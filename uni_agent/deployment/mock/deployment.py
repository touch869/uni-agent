"""Mock sandbox deployment for performance testing.

The LLM (vLLM replicas + router) runs for real; only the sandbox
(docker/swe-rex bash execution) is stubbed. ``MockRuntime`` implements
swerex's ``AbstractRuntime`` so the entire production code path
(``AgentEnv`` -> tools install -> ``run_action``) runs unmodified -- only
the leaf bash execution returns a representative canned observation.

Built test-first. Two responsibilities so far: the command router
(``_route``) and observation rendering with reproducible seeding.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from pathlib import Path

import yaml
from swerex.deployment.abstract import AbstractDeployment
from swerex.exceptions import CommandTimeoutError, DeploymentNotStartedError
from swerex.runtime.abstract import (
    AbstractRuntime,
    BashAction,
    BashInterruptAction,
    Observation,
)

from uni_agent.async_logging import get_logger

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
# Patterns checked in order; first match wins. The command string is the bash
# that ``tools_manager.get_tool_bash_command`` emits (see findings.md).
#
# Install-phase commands (which/export/chmod/mkdir/pip) must route to
# ``install`` so they no-op successfully and ``AgentEnv.install_tools`` passes.
_ROUTE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("finish", re.compile(r"""echo\s+['"]<<<Finished>>>['"]""")),
    ("install", re.compile(r"^(which|export|chmod|mkdir|pip|pip3)\b")),
    ("install", re.compile(r"\bpip3?\s+install\b")),
    ("install", re.compile(r"\bpython\d?\s+-m\s+pip\b")),
    ("editor:view", re.compile(r"^str_replace_editor\b.*--command\s+view\b")),
    ("editor:create", re.compile(r"^str_replace_editor\b.*--command\s+create\b")),
    ("editor:str_replace", re.compile(r"^str_replace_editor\b.*--command\s+str_replace\b")),
    ("editor:insert", re.compile(r"^str_replace_editor\b.*--command\s+insert\b")),
    ("editor:undo_edit", re.compile(r"^str_replace_editor\b.*--command\s+undo_edit\b")),
    ("test_output", re.compile(r"^(\S*python\S*\s+-m\s+pytest\b|^pytest\b)")),
    ("python_script", re.compile(r"^python\d?\s")),
    ("listing", re.compile(r"^(find|ls)\b")),
    ("search", re.compile(r"^grep\b")),
    ("file_view", re.compile(r"^(cat|head|tail)\b")),
]


# ---------------------------------------------------------------------------
# Template pool
# ---------------------------------------------------------------------------
# The single source of truth for observation templates is the bundled YAML
# (observations.yaml). Weights are hand-tuned to a realistic success:failure
# ratio (~8:2) and length spread. Text structure follows real SWE-bench samples.
#
# finish / install are fixed single outputs (not sampled from the pool).

_FINISH_OUTPUT = "<<<Finished>>>"

# Bundled YAML. Resolved relative to this file so it works regardless of CWD.
_DEFAULT_TEMPLATES_PATH = Path(__file__).parent / "observations.yaml"

# Module-level cache for the bundled default pool: it is immutable after load,
# so the (frequently-constructed) MockRuntime reuses one parse across all runs.
_DEFAULT_POOL: dict[str, list[tuple[int, str]]] | None = None


def load_templates(path: str | None = None) -> dict[str, list[tuple[int, str]]]:
    """Load the observation template pool from YAML.

    Schema: ``route_key -> [{weight: int, text: str}, ...]``. ``path=None`` loads
    the bundled default (cached after first read). Raises ``OSError`` or
    ``yaml.YAMLError`` if the file is missing or corrupt -- there is no in-code
    fallback; YAML is the single source of truth.
    """
    global _DEFAULT_POOL
    if path is None:
        if _DEFAULT_POOL is None:
            _DEFAULT_POOL = _parse_templates(_DEFAULT_TEMPLATES_PATH)
        return _DEFAULT_POOL
    return _parse_templates(Path(path))


def _parse_templates(path: Path) -> dict[str, list[tuple[int, str]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    pool: dict[str, list[tuple[int, str]]] = {}
    for key, entries in (raw or {}).items():
        pool[key] = [(int(e["weight"]), str(e["text"])) for e in entries]
    return pool


def _scale_text(text: str, scale: float) -> str:
    """Stretch (or shrink) ``text`` toward ``len(text) * scale`` chars.

    Grown by repeating a clean newline-terminated block; shrunk by truncating
    on a line boundary. A no-op at scale == 1.0.
    """
    if scale <= 0:
        return ""
    target = int(len(text) * scale)
    if target <= len(text):
        return text[:target].rsplit("\n", 1)[0]
    if not text:
        return text
    block = text if text.endswith("\n") else text + "\n"
    reps = max(1, target // max(1, len(block)) + 1)
    grown = (block * reps)[:target]
    if not grown.endswith("\n"):
        grown = grown.rsplit("\n", 1)[0]
    return grown


class MockRuntime(AbstractRuntime):
    """CPU-only runtime that returns canned observations by routing the
    command string. Subclassed from ``AbstractRuntime`` for protocol
    compatibility; no real session is created."""

    def __init__(
        self,
        run_id: str = "mock",
        *,
        seed: int | None = None,
        observation_scale: float = 1.0,
        templates: dict[str, list[tuple[int, str]]] | None = None,
        templates_path: str | None = None,
        timeout: "TimeoutSimConfig | None" = None,
        terminal_dead: "TerminalDeadConfig | None" = None,
    ) -> None:
        from .config import TerminalDeadConfig, TimeoutSimConfig

        self.run_id = run_id
        self._seed = seed
        self.observation_scale = observation_scale
        # Precedence: explicit dict > explicit yaml path > bundled default yaml.
        if templates is not None:
            self._templates = templates
        elif templates_path is not None:
            self._templates = load_templates(templates_path)
        else:
            self._templates = load_templates()
        self._timeout_cfg = timeout or TimeoutSimConfig()
        self._terminal_dead_cfg = terminal_dead or TerminalDeadConfig()
        # Failure state: once the terminal "dies" it stays dead -- every later
        # run_in_session (including env.py's liveness probe) returns no marker,
        # which is what makes env.py raise TerminalNotAliveError.
        self._dead = False
        # One independent RNG stream per route key, derived from
        # (global_seed, route_key). Same seed -> same per-key streams ->
        # reproducible sampling, without one command type's draws perturbing
        # another's. Failure modes get their own dedicated streams so timeout
        # and terminal_dead patterns are mutually reproducible/independent.
        self._rngs: dict[str, random.Random] = {}
        # Pre-unpack the (immutable) pool into parallel weight/text lists so
        # _render does a dict lookup instead of rebuilding lists every call.
        self._weights: dict[str, tuple[int, ...]] = {}
        self._texts: dict[str, tuple[str, ...]] = {}
        for key, pool in self._templates.items():
            self._rngs[key] = random.Random(self._derive_seed(key))
            if pool:
                self._weights[key], self._texts[key] = zip(*pool)
            else:
                self._weights[key], self._texts[key] = (), ()
        self._timeout_rng = random.Random(self._derive_seed("__timeout__"))
        self._dead_rng = random.Random(self._derive_seed("__terminal_dead__"))

    def _derive_seed(self, key: str) -> int | None:
        if self._seed is None:
            return None
        digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def _route(self, command: str) -> str:
        """Map a bash command string to a route key (a template group).

        Pure function of ``command``: same command always yields the same
        key. Order matters -- ``python -m pip install`` must hit ``install``
        before ``python_script``; ``str_replace_editor --command`` variants
        are split by subcommand.
        """
        stripped = command.strip()
        for key, pattern in _ROUTE_RULES:
            if pattern.search(stripped):
                return key
        return "default"

    def _render(self, route_key: str) -> str:
        """Sample a representative observation for ``route_key``.

        Fixed keys (finish/install) return constants; others sample by weight
        from the template pool, then apply ``observation_scale``.
        """
        if route_key == "finish":
            # The finish signal MUST be exact (not scaled) so the agent loop's
            # ``<<<Finished>>>`` detection fires correctly.
            return _FINISH_OUTPUT
        if route_key == "install":
            return ""
        # Fall back to the default pool for an unknown route key (e.g. an
        # editor subcommand we didn't model) rather than crash.
        key = route_key if route_key in self._texts else "default"
        texts = self._texts.get(key)
        if not texts:
            # Even the default pool is missing/empty -- emit a bare observation.
            return _scale_text("", self.observation_scale)
        rng = self._rngs.get(key)
        if rng is None:
            # A route key with no prebuilt stream (templates overridden to add
            # a new key): build one and cache it so subsequent calls advance the
            # RNG state rather than restarting from the same derived seed.
            rng = random.Random(self._derive_seed(key))
            self._rngs[key] = rng
        text = rng.choices(texts, weights=self._weights.get(key), k=1)[0]
        return _scale_text(text, self.observation_scale)

    async def run_in_session(self, action) -> Observation:
        if isinstance(action, BashInterruptAction):
            # A dead terminal cannot accept an interrupt either -- this is what
            # pushes env.py out of its interrupt_session() happy path and into
            # the liveness-probe branch that raises TerminalNotAliveError.
            if self._dead:
                raise CommandTimeoutError("mock: terminal dead, interrupt unresponsive")
            return Observation(output="", exit_code=130)
        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")
        route_key = self._route(action.command)

        # Once the terminal is dead, every call (including env.py's liveness
        # probe ``echo 'terminal still alive'``) returns no marker -> env.py
        # concludes the terminal is dead and raises TerminalNotAliveError.
        if self._dead:
            return Observation(output="", exit_code=1)

        # Failure simulation only applies to real tool commands, not the
        # install/finish phase (which must stay reliable so install_tools
        # passes and submit terminates cleanly).
        if route_key not in ("install", "finish"):
            # terminal_dead: model it as a hung command. Time out (which makes
            # env.py probe liveness), then stay dead so the probe fails.
            if self._terminal_dead_cfg.enabled and self._dead_rng.random() < self._terminal_dead_cfg.probability:
                self._dead = True
                raise CommandTimeoutError("mock: terminal died (command hung)")
            # timeout: recoverable. Sleep to mimic real bash wall-clock, then
            # raise so env.py decrements timeout_budget.
            if self._timeout_cfg.enabled and self._timeout_rng.random() < self._timeout_cfg.probability:
                if self._timeout_cfg.delay_seconds > 0:
                    await asyncio.sleep(self._timeout_cfg.delay_seconds)
                raise CommandTimeoutError("mock: simulated command timeout")

        output = self._render(route_key)
        return Observation(output=output, exit_code=0)


# --- AbstractRuntime protocol stubs (filled by later TDD cycles) -------------
# All are no-op stubs returning empty/default responses. Generated from a single
# table so adding/changing one is a one-line edit, not eight. Deferred-importing
# the response class inside the stub keeps swerex as a runtime-only dependency
# for these (uncalled) methods.
#
# ABCMeta computes ``__abstractmethods__`` once at class creation; setattr-ing
# concrete methods afterward does not clear it, so we drop the set explicitly to
# let MockRuntime instantiate (every abstract method is in fact provided below).
_RUNTIME_STUBS: dict[str, tuple[str, dict]] = {
    "create_session": ("CreateBashSessionResponse", {}),
    "execute": ("CommandResponse", {"stdout": "", "stderr": "", "exit_code": 0}),
    "upload": ("UploadResponse", {}),
    "read_file": ("ReadFileResponse", {"content": ""}),
    "write_file": ("WriteFileResponse", {}),
    "is_alive": ("IsAliveResponse", {"is_alive": True}),
    "close_session": ("CloseBashSessionResponse", {}),
    "close": ("CloseResponse", {}),
}


def _make_runtime_stub(name: str, response_attr: str, kwargs: dict):
    async def _stub(self, *args, **_kwargs):  # pragma: no cover - stub
        from swerex.runtime import abstract as _abstract

        return getattr(_abstract, response_attr)(**kwargs)

    _stub.__name__ = name
    _stub.__qualname__ = f"MockRuntime.{name}"
    return _stub


for _name, (_resp, _kwargs) in _RUNTIME_STUBS.items():
    setattr(MockRuntime, _name, _make_runtime_stub(_name, _resp, _kwargs))
MockRuntime.__abstractmethods__ = frozenset()


class MockDeployment(AbstractDeployment):
    """Deployment wrapper around :class:`MockRuntime`.

    Subclasses swerex's ``AbstractDeployment`` (like every other deployment) so
    ``AgentEnv`` drives it exactly like docker/modal/etc. ``start()`` does
    nothing real -- no process, no docker, no swe-rex -- it just marks the
    runtime ready. The configured seed/scale flow into the MockRuntime so
    sampling is reproducible before any command runs.
    """

    def __init__(
        self,
        run_id: str,
        *,
        seed: int | None = None,
        observation_scale: float = 1.0,
        timeout: TimeoutSimConfig | None = None,
        terminal_dead: TerminalDeadConfig | None = None,
    ) -> None:
        self.logger = get_logger("mock-deployment", run_id)
        self.run_id = run_id
        self._runtime = MockRuntime(
            run_id=run_id,
            seed=seed,
            observation_scale=observation_scale,
            timeout=timeout,
            terminal_dead=terminal_dead,
        )
        self._started = False

    @classmethod
    def from_config(cls, config, run_id: str | None = None) -> "MockDeployment":
        return cls(
            run_id=run_id or "mock",
            seed=config.seed,
            observation_scale=config.observation_scale,
            timeout=config.timeout,
            terminal_dead=config.terminal_dead,
        )

    def add_hook(self, hook):  # pragma: no cover - protocol no-op
        pass

    async def start(self, max_retries: int = 5) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def is_alive(self, *, timeout=None):
        from swerex.runtime.abstract import IsAliveResponse

        return IsAliveResponse(is_alive=self._started)

    @property
    def runtime(self) -> MockRuntime:
        if not self._started:
            raise DeploymentNotStartedError()
        return self._runtime

