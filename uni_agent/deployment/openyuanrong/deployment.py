"""OpenYuanRong/AKernel remote sandbox deployment."""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from pathlib import Path
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import CommandTimeoutError, DeploymentNotStartedError
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseBashSessionResponse,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateBashSessionRequest,
    CreateBashSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import OpenYuanRongDeploymentConfig


def _configure_credentials() -> None:
    server = os.getenv("AKERNEL_SERVER_ADDRESS") or os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("AKERNEL_TOKEN") or os.getenv("OPENYUANRONG_TOKEN")
    if not server or not token:
        raise ValueError(
            "AKERNEL_SERVER_ADDRESS/OPENYUANRONG_SERVER_ADDRESS and "
            "AKERNEL_TOKEN/OPENYUANRONG_TOKEN must be set"
        )
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token
    os.environ["TUNNEL_SSL_VERIFY"] = os.getenv(
        "AKERNEL_TUNNEL_SSL_VERIFY",
        os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", "0"),
    )


class OpenYuanRongRuntime(AbstractRuntime):
    """Adapt AKernel's persistent shell and filesystem to SWE-ReX runtime APIs."""

    def __init__(self, sandbox: Any, run_id: str):
        self.sandbox = sandbox
        self.logger = get_logger("openyuanrong-runtime", run_id)
        self._shells: dict[str, Any] = {}

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        del timeout
        try:
            alive = await asyncio.to_thread(self.sandbox.is_running)
            return IsAliveResponse(is_alive=bool(alive))
        except Exception as exc:
            return IsAliveResponse(is_alive=False, message=str(exc))

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        shell = await self.sandbox.shells.create(
            shell="/bin/bash",
            timeout=max(int(request.startup_timeout or 60), 1),
        )
        self._shells[request.session] = shell
        if request.startup_source:
            sources = " && ".join(f"source {shlex.quote(path)}" for path in request.startup_source)
            await shell.run(f"{sources} || true", timeout=max(int(request.startup_timeout or 60), 1))
        return CreateBashSessionResponse()

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        shell = self._shells.pop(request.session, None)
        if shell is not None:
            await shell.kill()
        return CloseBashSessionResponse()

    async def run_in_session(self, action: Action) -> Observation:
        shell = self._shells.get(action.session)
        if shell is None:
            raise DeploymentNotStartedError(f"Shell session {action.session!r} is not initialized")
        if isinstance(action, BashInterruptAction):
            # AKernel Shell.run interrupts the command itself when its timeout expires.
            return BashObservation(output="", exit_code=130)
        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")
        timeout = max(int(action.timeout or 60), 1)
        result = await shell.run(action.command, timeout=timeout)
        if result.exit_code == -1 and "timed out" in (result.stderr or "").lower():
            raise CommandTimeoutError(result.stderr)
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        return BashObservation(output=output, exit_code=result.exit_code)

    async def execute(self, command: Command) -> CommandResponse:
        if isinstance(command.command, list):
            cmd = shlex.join(str(part) for part in command.command)
        else:
            cmd = command.command
        result = await asyncio.to_thread(
            self.sandbox.commands.run,
            cmd,
            timeout=max(int(command.timeout or 60), 1),
            envs=command.env,
            cwd=command.cwd,
        )
        return CommandResponse(stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        content = await asyncio.to_thread(self.sandbox.files.read, request.path)
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        await asyncio.to_thread(self.sandbox.files.write, request.path, request.content)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        await asyncio.to_thread(
            self.sandbox.files.copy_from_local,
            request.source_path,
            request.target_path,
        )
        return UploadResponse()

    async def close(self) -> CloseResponse:
        for shell in list(self._shells.values()):
            await shell.kill()
        self._shells.clear()
        return CloseResponse()


class OpenYuanRongDeployment(AbstractDeployment):
    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = OpenYuanRongDeploymentConfig(**kwargs)
        self.logger = get_logger("openyuanrong-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._sandbox: Any | None = None
        self._runtime: OpenYuanRongRuntime | None = None

    @classmethod
    def from_config(cls, config: OpenYuanRongDeploymentConfig, run_id: str | None = None) -> Self:
        return cls(run_id=run_id or str(uuid.uuid4()), **config.model_dump())

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self, max_retries: int = 5) -> None:
        _configure_credentials()
        from akernel_sdk import Sandbox

        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                self._hooks.on_custom_step("Creating OpenYuanRong sandbox")
                self.logger.info(f"Creating OpenYuanRong sandbox with image={self._config.image}")
                kwargs = {
                    "image": self._config.image,
                    "cpu": self._config.cpu,
                    "memory": self._config.memory,
                    "cpu_limit": self._config.cpu_limit,
                    "mem_limit": self._config.mem_limit,
                    "idle_timeout": self._config.idle_timeout,
                    **self._config.sandbox_kwargs,
                }
                if self._config.name_prefix:
                    kwargs["name"] = f"{self._config.name_prefix}{uuid.uuid4().hex[:8]}"
                self._sandbox = await asyncio.to_thread(Sandbox, **kwargs)
                self._runtime = OpenYuanRongRuntime(self._sandbox, self.run_id)
                await self._runtime.create_session(
                    CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
                )
                self.logger.info(f"OpenYuanRong sandbox created: {self._sandbox.sandbox_id}")
                return
            except Exception as exc:
                last_error = exc
                self.logger.error(f"Failed to create OpenYuanRong sandbox: {exc}")
                await self.stop()
                sandbox_name = kwargs.get("name")
                if sandbox_name:
                    try:
                        await asyncio.to_thread(Sandbox.delete, sandbox_name)
                    except Exception as cleanup_exc:
                        self.logger.warning(
                            f"Failed to delete OpenYuanRong sandbox {sandbox_name} after create error: {cleanup_exc}"
                        )
                if retry < max_retries - 1:
                    await asyncio.sleep(min(30, 2**retry))
        raise RuntimeError(f"Failed to create OpenYuanRong sandbox after {max_retries} retries") from last_error

    async def stop(self) -> None:
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None
        if self._sandbox is not None:
            sandbox = self._sandbox
            self._sandbox = None
            try:
                if await asyncio.to_thread(sandbox.is_running):
                    await asyncio.to_thread(sandbox.kill)
            except Exception as exc:
                self.logger.warning(f"Failed to kill OpenYuanRong sandbox: {exc}")

    async def copy_to_container(self, src: Path, tgt: Path):
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        await self._runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self._runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    @property
    def runtime(self) -> OpenYuanRongRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return self._runtime

