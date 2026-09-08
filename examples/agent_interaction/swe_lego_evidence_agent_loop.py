"""SWE-Lego-only agent loop that persists evidence before the runtime is destroyed."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError
from swerex.runtime.abstract import BashAction

from uni_agent.agent_loop import UniAgentLoop
from uni_agent.async_logging import add_file_handler, cleanup_handlers, get_logger
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.interaction.env import ActionIncorrectSyntaxError, ActionTimeoutError
from uni_agent.reward.registry import register_reward_spec
from uni_agent.reward.swe_bench import SWEBenchRewardSpec, _make_eval_script_list
from uni_agent.utils import simple_timer
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from examples.agent_interaction.swe_lego_tool_parser import SWELegoToolParser


_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(https?://[^/@:\s]+:)[^/@\s]+(@)"),
]


def _redact_text(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 2:
            value = pattern.sub(r"\1<redacted>\2", value)
        else:
            value = pattern.sub(r"\1<redacted>", value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def _split_streams(output: str) -> tuple[str, str]:
    """Extract streams emitted by Uni-Agent's execute_bash wrapper when present."""
    if "[STDOUT]" not in output:
        return output, ""
    stdout = output.split("[STDOUT]", 1)[1]
    stderr = ""
    if "[STDERR]" in stdout:
        stdout, stderr = stdout.split("[STDERR]", 1)
    return stdout.strip(), stderr.strip()


class EvidenceAgentEnv(AgentEnv):
    """AgentEnv variant that retains the real SWE-ReX exit code for every action."""

    def __init__(self, run_id: str, env_config: AgentEnvConfig):
        super().__init__(run_id=run_id, env_config=env_config)
        self.action_records: list[dict[str, Any]] = []

    async def run_action(self, action_cmd: str, action_timeout: int, max_observation_length: int = 12_000) -> str:
        started = time.perf_counter()
        try:
            result = await self.deployment.runtime.run_in_session(
                BashAction(command=action_cmd, timeout=action_timeout, check="silent")
            )
            raw_output = result.output or ""
            stdout, stderr = _split_streams(raw_output)
            self.action_records.append(
                {
                    "action": action_cmd,
                    "stdout": stdout,
                    "stderr": stderr,
                    "combined_output": raw_output,
                    "exit_code": result.exit_code,
                    "execution_time_s": time.perf_counter() - started,
                    "status": "ok" if result.exit_code in (0, None) else "nonzero_exit",
                }
            )
            cleaned = re.sub(r"\x1b\[[0-9;]*m|\r", "", raw_output).strip()
            if not cleaned:
                return "Your command ran successfully and did not produce any output."
            if len(cleaned) > max_observation_length:
                cleaned = cleaned[:max_observation_length] + "<response clipped>"
            return f"Observation:\n{cleaned}"
        except CommandTimeoutError as exc:
            self.action_records.append(
                {
                    "action": action_cmd,
                    "stdout": "",
                    "stderr": str(exc),
                    "combined_output": str(exc),
                    "exit_code": None,
                    "execution_time_s": time.perf_counter() - started,
                    "status": "timeout",
                }
            )
            try:
                await self.interrupt_session()
            except Exception:
                pass
            raise ActionTimeoutError(f"Command exceeded {action_timeout} seconds and was interrupted: {action_cmd}") from None
        except BashIncorrectSyntaxError as exc:
            extra = getattr(exc, "extra_info", {}) or {}
            self.action_records.append(
                {
                    "action": action_cmd,
                    "stdout": extra.get("bash_stdout", ""),
                    "stderr": extra.get("bash_stderr", str(exc)),
                    "combined_output": str(exc),
                    "exit_code": None,
                    "execution_time_s": time.perf_counter() - started,
                    "status": "syntax_error",
                }
            )
            raise ActionIncorrectSyntaxError(str(exc)) from None


@register_reward_spec("swe_bench_evidence")
class EvidenceSWEBenchRewardSpec(SWEBenchRewardSpec):
    """SWE-Bench reward that retains the exact evaluator script/output/exit code."""

    async def compute_reward(self, **kwargs):
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        instance = self.metadata
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance.get("version")]
        commands = _make_eval_script_list(
            instance=instance,
            specs=specs,
            env_name="testbed",
            repo_directory="/testbed",
            base_commit=instance["base_commit"],
            test_patch=instance["test_patch"],
        )
        script = "\n".join(["#!/bin/bash", "set -uxo pipefail", *commands]) + "\n"
        script_path = Path(f"/tmp/eval_script_evidence_{uuid.uuid4()}.sh")
        await self.env.write_file(script_path, script)
        started = time.perf_counter()
        result: dict[str, Any] = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_script": script,
            "eval_command": f"bash {script_path} 2>&1",
            "eval_exit_code": None,
            "eval_output": "",
            "eval_report": None,
            "resolved": False,
        }
        try:
            obs = await self.env.deployment.runtime.run_in_session(
                BashAction(command=result["eval_command"], timeout=self.eval_timeout, check="silent")
            )
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", obs.output or "")
            result.update(
                eval_completed=True,
                eval_execution_time=time.perf_counter() - started,
                eval_exit_code=obs.exit_code,
                eval_output=output,
            )
            report = self._get_eval_report(output)
            result["eval_report"] = report
            result["resolved"] = bool(report.get("resolved"))
            self.logger.info(f"Eval report: {report}")
        except Exception as exc:
            result["eval_execution_time"] = time.perf_counter() - started
            result["eval_error"] = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"Failed to evaluate with evidence: {exc}")
        self.last_evaluator_evidence = result
        return result["resolved"], result


class SWELegoEvidenceAgentLoop(UniAgentLoop):
    """Independent loop used only by parallel_infer_swe_lego_evidence.py."""

    def _init_env(self, config_dict: dict) -> AgentEnv:
        return EvidenceAgentEnv(run_id=self.run_id, env_config=AgentEnvConfig(**config_dict))

    async def _capture_repo(self, label: str) -> dict[str, Any]:
        commands = {
            "status": "cd /testbed && git status --short",
            "diff": "cd /testbed && git diff --no-ext-diff --no-color --binary HEAD",
            "cached_diff": "cd /testbed && git diff --cached --no-ext-diff --no-color --binary HEAD",
            "recent_history": "history 2>/dev/null | tail -200 || true",
        }
        captured: dict[str, Any] = {"label": label}
        for key, command in commands.items():
            try:
                obs = await self.env.deployment.runtime.run_in_session(
                    BashAction(command=command, timeout=60, check="silent")
                )
                captured[key] = {"command": command, "output": obs.output or "", "exit_code": obs.exit_code}
            except Exception as exc:
                captured[key] = {"command": command, "error": f"{type(exc).__name__}: {exc}"}
        return captured

    async def _parse_steps(self, trajectory) -> list[dict[str, Any]]:
        action_iter = iter(getattr(self.env, "action_records", []))
        parsed_steps = []
        for step in trajectory:
            item = _jsonable(step)
            parsed_calls = []
            parse_error = None
            if step.response:
                try:
                    _, calls = await self.tools_manager.parse_action(step.response)
                    parsed_calls = [_jsonable(call) for call in calls]
                    item["tool_parser_adapter"] = _jsonable(getattr(self.tools_manager._tool_parser, "last_event", None))
                except Exception as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"
            item["parsed_tool_calls"] = parsed_calls
            item["tool_call_parse_error"] = parse_error
            for tool_result in item.get("tool_results", []):
                tool_result["runtime_evidence"] = _jsonable(next(action_iter, None))
            parsed_steps.append(item)
        return parsed_steps

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        wall_started = time.perf_counter()
        config_dict = self._init_config(sampling_params, **kwargs)
        self.mask_abnormal_exit_traj = config_dict.get("mask_abnormal_exit_traj", False)
        concurrency = config_dict.get("concurrency", 512)
        workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        if UniAgentLoop._semaphore is None:
            UniAgentLoop._semaphore = asyncio.Semaphore(max(concurrency // workers, 1))

        self.run_id = str(uuid.uuid4())
        self.logger = get_logger("swe-lego-evidence", run_id=self.run_id)
        self.chat_model = self._init_chat_model(config_dict["model"])
        self.tools_manager = self._init_tools_manager(
            config_dict["tools"], parser=config_dict.get("tool_parser", "hermes")
        )
        if config_dict.get("tool_parser", "hermes") == "hermes":
            self.tools_manager._tool_parser = SWELegoToolParser()
        self.skills_manager = self._init_skills_manager(config_dict.get("skills"))
        self.env = self._init_env(config_dict["env"])
        self.output_dir = Path(config_dict["log_dir"]) / self.run_id
        self.interaction = self._init_interaction(config_dict, kwargs)
        reward_config = {**config_dict["reward"], "run_id": self.run_id, "env": self.env}
        from uni_agent.reward import load_reward_spec

        self.reward_spec = load_reward_spec(reward_config)
        evidence_meta = config_dict.get("evidence", {})

        async with self._semaphore:
            add_file_handler(self.output_dir / "run.log", self.run_id)
            interaction_result = None
            evidence: dict[str, Any] = {
                "schema_version": 1,
                "run_name": evidence_meta.get("run_name", os.getenv("SWE_LEGO_RUN_NAME", "unknown")),
                "rollout_index": evidence_meta.get("rollout_index"),
                "run_id": self.run_id,
                "model_path": self.config.actor_rollout_ref.model.path,
                "sampling_params": sampling_params,
            }
            try:
                await self.env.start()
                self.chat_model.set_tools_schemas(self.tools_manager.tools_schemas)
                await self.env.install_tools(self.tools_manager.tools)
                problem_statement = (
                    config_dict.get("reward", {}).get("metadata", {}).get("problem_statement", "")
                )
                prompt_adapter = {
                    "name": "openhands_concise_v1",
                    "preserves_full_problem_statement": True,
                    "turn_budget": self.interaction.max_turns,
                }
                adapted_user_prompt = (
                    "<uploaded_files>\n/testbed\n</uploaded_files>\n"
                    "Fix the following issue in the repository at /testbed.\n\n"
                    f"<issue_description>\n{problem_statement}\n</issue_description>\n\n"
                    f"You have at most {self.interaction.max_turns} tool-call turns. Inspect only the "
                    "most relevant code, make the minimal change to non-test production files, run "
                    "the focused reproduction or unit tests, then call submit. Do not modify existing "
                    "tests and do not spend turns restating the issue or workflow."
                )
                user_message = next(
                    (message for message in self.interaction.messages if message.get("role") == "user"), None
                )
                if user_message is None:
                    self.interaction.messages.append({"role": "user", "content": adapted_user_prompt})
                else:
                    user_message["content"] = adapted_user_prompt
                adapter_instruction = (
                    "\n\nExecution constraints for this run:\n"
                    f"- You have at most {self.interaction.max_turns} tool-call turns.\n"
                    "- Use only exact <tool_call> JSON from the tool schema; do not emit "
                    "<execute_bash>, <functionCall>, <response>, or HTML tags.\n"
                    "- Use grep or bounded view_range; never view a whole large file. After reproducing, "
                    "immediately edit production code without browsing tests first; run tests by turn 9.\n"
                    "- Do not repeat the issue or narrate the numbered workflow; act with tools."
                )
                system_message = next(
                    (message for message in self.interaction.messages if message.get("role") == "system"), None
                )
                if system_message is None:
                    self.interaction.messages.insert(0, {"role": "system", "content": adapter_instruction.strip()})
                else:
                    system_message["content"] = (system_message.get("content") or "") + adapter_instruction
                initial_messages = _jsonable(list(self.interaction.messages))
                rendered_prompt = self.tokenizer.apply_chat_template(
                    self.interaction.messages,
                    tools=self.tools_manager.tools_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompt_tokens = len(
                    self.tokenizer.apply_chat_template(
                        self.interaction.messages,
                        tools=self.tools_manager.tools_schemas,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                )
                interaction_result = await self.interaction.run()
                interaction_result["metrics"] = dict(interaction_result.get("rollout_cache", {}).get("metrics", {}))
                parser_adapter_events = list(self.tools_manager._tool_parser.events)
                pre_eval_repo = await self._capture_repo("before_evaluator")
                reward_score, evaluator = await self.reward_spec.compute_reward(interaction_result=interaction_result)
                interaction_result["reward_score"] = reward_score
                post_eval_repo = await self._capture_repo("after_evaluator")
                steps = await self._parse_steps(interaction_result["trajectory"])
                metadata = config_dict.get("reward", {}).get("metadata", {})
                final_step = steps[-1] if steps else {}
                test_steps = []
                test_pattern = re.compile(r"(^|\s)(pytest|tox|nox|unittest|test|runtests\.py|manage\.py\s+test)(\s|$)")
                for step in steps:
                    for result in step.get("tool_results", []):
                        call = result.get("runtime_evidence") or {}
                        action = call.get("action", "")
                        if test_pattern.search(action):
                            test_steps.append({"step_idx": step.get("step_idx"), **result})
                final_patch = (pre_eval_repo.get("diff") or {}).get("output", "")
                cached_diff = (pre_eval_repo.get("cached_diff") or {}).get("output", "")
                if cached_diff and cached_diff not in final_patch:
                    final_patch += "\n" + cached_diff
                evidence.update(
                    instance_id=metadata.get("instance_id"),
                    image=config_dict.get("env", {}).get("deployment", {}).get("image"),
                    repository=metadata.get("repo"),
                    problem_statement=metadata.get("problem_statement"),
                    initial_messages=initial_messages,
                    system_prompt=[m for m in initial_messages if m.get("role") == "system"],
                    user_prompt=[m for m in initial_messages if m.get("role") == "user"],
                    applied_chat_template=rendered_prompt,
                    prompt_adapter=prompt_adapter,
                    prompt_token_count=prompt_tokens,
                    tool_schemas=self.tools_manager.tools_schemas,
                    tool_parser_adapter_events=parser_adapter_events,
                    trajectory=steps,
                    messages=interaction_result["messages"],
                    termination_reason=final_step.get("exit_reason"),
                    actual_turns=len(steps),
                    hit_max_turns=final_step.get("exit_reason") == "max_step_limit",
                    hit_token_limit=final_step.get("exit_reason") == "token_limit",
                    hit_timeout=any(
                        r.get("status") == "timeout" for s in steps for r in s.get("tool_results", [])
                    ),
                    final_answer=final_step.get("thought") or final_step.get("response", ""),
                    final_patch=final_patch,
                    patch_empty=not bool(final_patch.strip()),
                    model_test_commands=test_steps,
                    repository_evidence={"before_evaluator": pre_eval_repo, "after_evaluator": post_eval_repo},
                    evaluator=evaluator,
                    rm_score=reward_score,
                    resolved=bool(evaluator.get("resolved")),
                    cache_metrics=interaction_result.get("metrics", {}),
                    generation_time_s=interaction_result.get("metrics", {}).get("generate_sequences"),
                    interaction_time_s=interaction_result.get("execution_time"),
                    wall_time_s=time.perf_counter() - wall_started,
                )
                self._save_interaction_result(interaction_result)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / "evidence.json").write_text(
                    json.dumps(_jsonable(evidence), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                output = await self.convert_to_agent_output(interaction_result)
            except Exception as exc:
                evidence["fatal_error"] = f"{type(exc).__name__}: {exc}"
                evidence["wall_time_s"] = time.perf_counter() - wall_started
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / "evidence.json").write_text(
                    json.dumps(_jsonable(evidence), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.logger.opt(exception=True).critical(f"Evidence agent loop failed: {exc}")
                output = await self._build_empty_agent_output(exit_reason="agent_loop_failed")
            finally:
                await self.env.close()
                cleanup_handlers(self.run_id)
            return output

    def _init_interaction(self, config_dict: dict, kwargs: dict):
        from uni_agent.interaction import AgentInteraction

        return AgentInteraction(
            run_id=self.run_id,
            env=self.env,
            model=self.chat_model,
            tools_manager=self.tools_manager,
            messages=list(kwargs["raw_prompt"]),
            skills_manager=self.skills_manager,
            **config_dict["interaction"],
        )
