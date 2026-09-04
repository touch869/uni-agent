import asyncio
import time
from typing import Literal

import orjson
from pydantic import BaseModel, Field

from uni_agent.async_logging import get_logger
from uni_agent.skills.manager import SkillsManager
from uni_agent.toolcall_spec import (
    DEFAULT_ELIGIBLE_TOOLS,
    TOOLCALL_METRIC_DEFAULTS,
    ObservationCache,
    canonical_tool_call,
    exact_tool_call_signature,
    finalize_toolcall_metrics,
    initialize_toolcall_metrics,
    is_eligible_tool_call,
)
from uni_agent.utils import auto_await, simple_timer

from .env import ActionIncorrectSyntaxError, ActionTimeoutError, AgentEnv, TerminalNotAliveError
from .model import AgentChatModel, MaxTokenExceededError
from .tool_parser import FunctionCallFormatError
from .tool_schemas import OpenAIFunctionToolCall
from .tools_manager import ToolsManager

ToolStatus = Literal["ok", "timeout", "syntax_error", "skipped"]


class ToolResult(BaseModel):
    """Per-tool-call result inside a single step. ``observation`` is the
    string sent back to the model as the ``role="tool"`` content (error
    text included).
    """

    tool_call_id: str
    name: str
    action: str = ""
    observation: str = ""
    status: ToolStatus
    execution_time: float | None = None


class StepOutput(BaseModel):
    step_idx: int

    response: str = ""
    thought: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_hash: str = ""
    response_hash: str = ""
    response_text_hash: str = ""
    hash_basis: str = ""
    generation_source: Literal["normal", "candidate"] | str = ""
    generation_wait_time_s: float = 0.0
    elapsed_time_s: float = 0.0


def fast_deepcopy(obj):
    return orjson.loads(orjson.dumps(obj))


class AgentInteraction:
    def __init__(
        self,
        run_id: str,
        env: AgentEnv,
        model: AgentChatModel,
        tools_manager: ToolsManager,
        messages: list[dict[str, str]],
        action_timeout: int = 60,
        timeout_budget: int = 3,
        max_turns: int = 50,
        max_format_retries_per_turn: int = 3,
        max_generation_tokens: int | None = None,
        toolcall_encoding_timeout: float = 10.0,
        toolcall_candidate_wait_timeout: float = 30.0,
        skills_manager: SkillsManager | None = None,
        chat_mode: bool = False,
        toolcall_speculation_enabled: bool = False,
        observation_cache: ObservationCache | None = None,
        toolcall_allowed_tools: frozenset[str] = DEFAULT_ELIGIBLE_TOOLS,
    ):
        """:param chat_mode: how to treat an assistant message with no
        tool calls. ``False`` (default, training / code-eval) raises
        ``format_error`` so the loop continues. ``True`` (long-running
        chat) marks the step ``turn_done`` so the caller can wait for
        the next user message.
        """
        self.env = env
        self.model = model
        self.tools_manager = tools_manager
        self.skills_manager = skills_manager
        self.messages = messages
        self.action_timeout = action_timeout
        self.timeout_budget = timeout_budget
        self.max_turns = max_turns
        if max_format_retries_per_turn < 0:
            raise ValueError("max_format_retries_per_turn must be non-negative")
        self.max_format_retries_per_turn = max_format_retries_per_turn
        if max_generation_tokens is not None and max_generation_tokens <= 0:
            raise ValueError("max_generation_tokens must be positive")
        self.max_generation_tokens = max_generation_tokens
        if toolcall_encoding_timeout <= 0:
            raise ValueError("toolcall_encoding_timeout must be positive")
        self.toolcall_encoding_timeout = toolcall_encoding_timeout
        if toolcall_candidate_wait_timeout <= 0:
            raise ValueError("toolcall_candidate_wait_timeout must be positive")
        self.toolcall_candidate_wait_timeout = toolcall_candidate_wait_timeout
        self.chat_mode = chat_mode
        self.toolcall_speculation_enabled = toolcall_speculation_enabled
        self.observation_cache = observation_cache if observation_cache is not None else ObservationCache()
        self.toolcall_allowed_tools = toolcall_allowed_tools
        self.run_id = run_id
        self.pending_generation = None
        self.logger = get_logger("interaction", run_id)

    def _discard_generated_tokens(self, token_count: int) -> None:
        if token_count <= 0:
            return
        for key in ("prompt_ids", "response_mask", "response_logprobs"):
            values = self.rollout_cache.get(key)
            if values is not None:
                del values[max(len(values) - token_count, 0) :]
        # This comparison model is dense, so routed_experts is always None.
        # Refuse to retain stale routing data if another model enables it.
        if self.rollout_cache.get("routed_experts") is not None:
            self.rollout_cache["routed_experts"] = None

    @staticmethod
    def _cancel_background_task(task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()

        def consume_result(completed_task: asyncio.Task) -> None:
            if completed_task.cancelled():
                return
            try:
                completed_task.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(consume_result)

    def inject_skills_manifest(self) -> None:
        """Append the skills manifest to the first system message.

        The manifest lists each discovered skill (name + description +
        path to its SKILL.md) so the model knows what is available and
        how to load it on demand. Skill *bodies* are not in the prompt --
        they live as real files on disk (read lazily, progressive
        disclosure).

        Call this exactly once, after ``AgentEnv.install_skills`` has
        populated ``runtime_paths``. The method is **not** idempotent --
        calling it twice will append the manifest twice. The single
        in-tree caller (``UniAgentLoop.run``) already enforces this.
        """
        if self.skills_manager is None:
            return
        manifest = self.skills_manager.build_manifest()
        if not manifest:
            return

        block = "\n\n" + manifest
        for msg in self.messages:
            if msg.get("role") == "system":
                content = msg.get("content") or ""
                msg["content"] = content + block
                return
        self.messages.insert(0, {"role": "system", "content": manifest})

    async def step(self, step_idx: int):
        """Run one model-call + tool-execution cycle.

        Outcome is reported at two levels:

        * **Tool**: per-call :class:`ToolResult` on ``step_output.tool_results``
          with ``status`` in ``{ok, timeout, syntax_error, skipped}``.
        * **Step**: ``step_output.exit_reason`` + ``done``:

          - terminal (``done=True``): ``finished``, ``turn_done``,
            ``token_limit``, ``terminal_dead``, ``timeout_budget_exhausted``.
          - non-terminal (``done=False``): ``completed``,
            ``completed_with_tool_errors``, ``format_error``.
          - set by :meth:`run`: ``max_step_limit``, ``unknown_error``.

        ``turn_done`` is gated on ``self.chat_mode`` (see ``__init__``).
        """
        # step index start from 1
        step_output = StepOutput(step_idx=step_idx)
        self.logger.info(f"{'=' * 25} STEP {step_idx} {'=' * 25}")

        # step 1: prepare template
        self.logger.info(f"🤖 MODEL INPUT\n{self.messages[-1]['content']}")

        # step 2: consume a committed candidate or generate normally.
        generation_started_at = time.perf_counter()
        try:
            pending = self.pending_generation
            self.pending_generation = None
            if pending is not None:
                model_output = pending.response
                tool_calls = pending.tool_calls
                rollout_cache = pending.rollout_cache
                generation_info = pending.generation_info
                generation_source = "candidate"
                self.logger.info("Consumed committed speculative candidate")
            else:
                fallback_ready_at = getattr(self, "_candidate_fallback_ready_at", None)
                if fallback_ready_at is not None:
                    metrics = self.rollout_cache.setdefault("metrics", {})
                    metrics["candidate_fallback_start_delay_ms"] += (time.perf_counter() - fallback_ready_at) * 1000
                    self._candidate_fallback_ready_at = None
                query_kwargs = {}
                if self.max_generation_tokens is not None:
                    sampling_params = dict(getattr(self.model, "sampling_params", {}) or {})
                    sampling_params["max_tokens"] = self.max_generation_tokens
                    query_kwargs["sampling_params"] = sampling_params
                model_output, tool_calls, rollout_cache, generation_info = await self.model.query(
                    messages=self.messages,
                    rollout_cache=self.rollout_cache,
                    **query_kwargs,
                )
                generation_source = "normal"
            step_output.response = model_output
            step_output.prompt_tokens = generation_info.get("prompt_tokens", 0)
            step_output.completion_tokens = generation_info.get("completion_tokens", 0)
            step_output.prompt_hash = generation_info.get("prompt_hash", "")
            step_output.response_hash = generation_info.get("response_hash", "")
            step_output.response_text_hash = generation_info.get("response_text_hash", "")
            step_output.hash_basis = generation_info.get("hash_basis", "")
            step_output.generation_source = generation_source
            self.logger.info(
                f"Prompt Tokens: {generation_info['prompt_tokens']}, "
                f"Completion Tokens: {generation_info['completion_tokens']}"
            )
            self.logger.debug(f"Model Output:\n{model_output}")
        except MaxTokenExceededError as e:
            step_output.generation_wait_time_s = time.perf_counter() - generation_started_at
            _msg = (
                f"[step{step_idx}] MaxTokenExceededError: "
                f"response_mask_len_before={len(self.rollout_cache.get('response_mask', []))} "
                f"prompt_ids_len={len(self.rollout_cache.get('prompt_ids', []))} "
                f"detail: {str(e)}"
            )
            self.logger.error("{}", _msg)
            step_output.exit_reason = "token_limit"
            step_output.done = True
            return step_output
        step_output.generation_wait_time_s = time.perf_counter() - generation_started_at

        # step 3: parse model response to actions
        self.rollout_cache = rollout_cache

        # Persist the assistant message in api-shape (with tool_calls)
        # so replay preserves the assistant<->tool linkage.
        assistant_msg: dict[str, object] = {"role": "assistant", "content": model_output}
        if tool_calls:
            normalize_ids = getattr(self.tools_manager, "normalize_structured_tool_call_ids", None)
            if normalize_ids is not None:
                tool_calls = normalize_ids(tool_calls, step_idx)
            assistant_msg["tool_calls"] = tool_calls
        self.messages.append(assistant_msg)

        try:
            if tool_calls:
                content, tool_calls = await self.tools_manager.parse_structured_action(
                    content=model_output,
                    tool_calls_data=tool_calls,
                    step_idx=step_idx,
                )
            else:
                content, tool_calls = await self.tools_manager.parse_action(
                    model_output=model_output,
                    step_idx=step_idx,
                )
            if not tool_calls and not self.chat_mode:
                raise FunctionCallFormatError("No function call found in the response.")
            if tool_calls:
                assistant_msg["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
        except FunctionCallFormatError as e:
            hit_generation_limit = (
                self.max_generation_tokens is not None
                and generation_info.get("completion_tokens", 0) >= self.max_generation_tokens
            )
            if hit_generation_limit:
                if self.messages and self.messages[-1] is assistant_msg:
                    self.messages.pop()
                self._discard_generated_tokens(generation_info.get("completion_tokens", 0))
            feedback = (
                f"Tool call rejected: {e}\n"
                "Return exactly one valid tool call and Do not explain outside it. "
                "Escape inner double quotes and newlines inside JSON string values."
            )
            if tool_calls:
                error_msgs: list[dict[str, object]] = [
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": feedback,
                    }
                    for tc in tool_calls
                ]
            else:
                # A missing call has no tool-call id to link.  A user-role
                # correction keeps the next retry well-formed and actionable.
                error_msgs = [{"role": "user", "content": feedback}]
            self.messages.extend(error_msgs)
            self.rollout_cache = await self.model.append_messages_to_rollout_cache(error_msgs, self.rollout_cache)
            step_output.exit_reason = "format_error"
            model_output_preview = "\n".join(model_output.splitlines()[:20])
            _msg = (
                f"Fail to parse thought and action from model output.\n"
                f"Error Message: {str(e)}\n"
                f"Model Output (first 20 lines): {model_output_preview}"
            )
            self.logger.error("{}", _msg)
            return step_output

        step_output.thought = content
        self.logger.info(f"💭 THOUGHT:\n{content}")

        # step 4: chat_mode-only end-of-turn (single-shot already raised above).
        if not tool_calls:
            step_output.done = True
            step_output.exit_reason = "turn_done"
            self.logger.info(f"💬 TURN DONE (no tool call): {model_output}")
            return step_output

        # step 5: run tools while an eligible candidate is generated.
        candidate = None
        candidate_key = None
        predicted_ids: list[int] = []
        candidate_base_context: list[int] = []
        candidate_started_at = 0.0
        metrics = self.rollout_cache.setdefault("metrics", {})
        initialize_toolcall_metrics(metrics)
        if getattr(self.model, "supports_toolcall_speculation", False):
            metrics["toolcall_lookup_attempts"] += 1
            try:
                candidate_key = canonical_tool_call(tool_calls[0], self.model.tool_schema_signature)
                candidate_call_signature = exact_tool_call_signature(tool_calls[0])
                cached = self.observation_cache.get(candidate_key)
            except (TypeError, ValueError, AttributeError):
                cached = None
            eligible = is_eligible_tool_call(
                enabled=self.toolcall_speculation_enabled,
                tool_calls=tool_calls,
                cache=self.observation_cache,
                cache_key=candidate_key,
                allowed_tools=self.toolcall_allowed_tools,
            )
            if eligible and cached is not None:
                metrics["canonical_call_matches"] += 1
                if cached.source_call_signature not in {None, candidate_call_signature}:
                    metrics["equivalent_tool_matches"] += 1
                predicted_message = {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].id,
                    "name": tool_calls[0].function.name,
                    "content": cached.content,
                }
                try:
                    predicted_ids = await asyncio.wait_for(
                        self.model.encode_tool_messages([predicted_message]),
                        timeout=self.toolcall_encoding_timeout,
                    )
                    candidate_base_context = list(self.rollout_cache["prompt_ids"])
                    candidate_cache = self.model.clone_rollout_cache(
                        self.rollout_cache,
                        candidate_id=f"{self.rollout_cache['request_id']}-candidate-step-{step_idx}",
                    )
                    candidate_started_at = time.perf_counter()
                    candidate_task = asyncio.create_task(
                        self.model.query_candidate(
                            candidate_cache,
                            predicted_ids,
                            sampling_params={
                                **(getattr(self.model, "sampling_params", {}) or {}),
                                **(
                                    {"max_tokens": self.max_generation_tokens}
                                    if self.max_generation_tokens is not None
                                    else {}
                                ),
                            },
                        )
                    )
                    candidate = {
                        "task": candidate_task,
                        "key": candidate_key,
                        "predicted_ids": predicted_ids,
                        "base_context": candidate_base_context,
                        "policy_version": getattr(self.model, "current_policy_version", None),
                        "sampling_signature": getattr(self.model, "sampling_signature", ""),
                        "tool_schema_signature": self.model.tool_schema_signature,
                        "has_no_environment_side_effect": True,
                        "started_at": candidate_started_at,
                    }
                    metrics["candidate_started"] += 1
                except Exception as exc:
                    self.logger.warning(f"Unable to start speculative candidate: {type(exc).__name__}: {exc}")
                    candidate = None

        tool_results: list[ToolResult] = []
        tool_messages: list[dict[str, object]] = []
        saw_finish = False
        terminal_dead = False

        with simple_timer("tool_calls", self.rollout_cache["metrics"]):
            for idx, tool_call in enumerate(tool_calls):
                tool_call: OpenAIFunctionToolCall  # type: ignore[no-redef]
                action_cmd = self.tools_manager.get_tool_bash_command(tool_call)
                self.logger.info(f"🎬 ACTION ({tool_call.function.name}):\n{action_cmd}")

                tool_t0 = time.perf_counter()
                status: ToolStatus
                try:
                    observation = await self.env.run_action(action_cmd, action_timeout=self.action_timeout)
                    status = "ok"
                    if tool_call.function.name in ("finish", "submit"):
                        saw_finish = True
                except ActionTimeoutError as e:
                    observation = str(e)
                    status = "timeout"
                    self.timeout_budget -= 1
                    self.logger.error(f"{observation} (timeout_budget left: {self.timeout_budget})")
                except ActionIncorrectSyntaxError as e:
                    observation = str(e)
                    status = "syntax_error"
                    self.logger.error(observation)
                except TerminalNotAliveError as e:
                    observation = str(e)
                    status = "skipped"
                    terminal_dead = True
                    self.logger.error(observation)
                elapsed = time.perf_counter() - tool_t0

                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        action=action_cmd,
                        observation=observation,
                        status=status,
                        execution_time=elapsed,
                    )
                )
                try:
                    result_key = canonical_tool_call(tool_call, self.model.tool_schema_signature)
                    self.observation_cache.put(
                        result_key,
                        observation,
                        status=status,
                        source_trajectory_id=self.run_id,
                        source_tool_name=tool_call.function.name,
                        source_call_signature=exact_tool_call_signature(tool_call),
                    )
                except (TypeError, ValueError, AttributeError):
                    pass

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": observation,
                    }
                )

                # On hard failure (dead session / budget out), synthesize
                # `skipped` results for remaining tool calls to keep the
                # assistant<->tool N:N invariant, then break.
                budget_exhausted = self.timeout_budget < 0
                if terminal_dead or budget_exhausted:
                    skipped_reason = (
                        "Skipped: the bash session died mid-step; no further tool calls ran."
                        if terminal_dead
                        else "Skipped: timeout budget exhausted mid-step; no further tool calls ran."
                    )
                    for remaining in tool_calls[idx + 1 :]:
                        tool_results.append(
                            ToolResult(
                                tool_call_id=remaining.id,
                                name=remaining.function.name,
                                action="",
                                observation=skipped_reason,
                                status="skipped",
                                execution_time=None,
                            )
                        )
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": remaining.id,
                                "name": remaining.function.name,
                                "content": skipped_reason,
                            }
                        )
                    break

        # step 6: validate candidate against the actual encoded observation.
        can_encode_context = getattr(self.model, "supports_toolcall_speculation", False) and hasattr(
            self.model, "encode_tool_messages"
        )
        actual_ids = None
        if can_encode_context:
            try:
                actual_ids = await asyncio.wait_for(
                    self.model.encode_tool_messages(tool_messages),
                    timeout=self.toolcall_encoding_timeout,
                )
            except TimeoutError:
                self.logger.warning("Timed out encoding actual tool messages; discarding speculative candidate.")
        actual_context = self.rollout_cache["prompt_ids"] + actual_ids if actual_ids is not None else None
        if candidate is not None:
            validation_started_at = time.perf_counter()
            metrics["candidate_tool_execution_ms"] += sum((tr.execution_time or 0.0) * 1000 for tr in tool_results)
            predicted_matches = candidate["predicted_ids"] == actual_ids
            tools_succeeded = all(tool_result.status == "ok" for tool_result in tool_results)
            session_active = not terminal_dead and self.timeout_budget >= 0 and not saw_finish
            candidate_output = None
            waited_after_tool_ms = 0.0
            wait_timed_out = False
            try:
                if predicted_matches and tools_succeeded and session_active:
                    if not candidate["task"].done():
                        wait_started_at = time.perf_counter()
                        done_tasks, _ = await asyncio.wait(
                            {candidate["task"]},
                            timeout=self.toolcall_candidate_wait_timeout,
                        )
                        waited_after_tool_ms = (time.perf_counter() - wait_started_at) * 1000
                        metrics["candidate_wait_after_tool_ms"] += waited_after_tool_ms
                        if done_tasks:
                            candidate_output = candidate["task"].result()
                        else:
                            wait_timed_out = True
                            metrics["candidate_wait_timeout_ms"] += waited_after_tool_ms
                            metrics["candidate_abort_wait_timeout"] += 1
                            metrics["candidate_cancel_requested"] += 1
                            metrics["candidate_cancelled"] += 1
                            self._cancel_background_task(candidate["task"])
                            self._candidate_fallback_ready_at = time.perf_counter()
                    else:
                        candidate_output = await candidate["task"]
                        metrics["candidate_completed_before_validation"] += 1
                elif not candidate["task"].done():
                    metrics["candidate_cancel_requested"] += 1
                    metrics["candidate_cancelled_elapsed_ms"] += (time.perf_counter() - candidate["started_at"]) * 1000
                    cancel_started_at = time.perf_counter()
                    metrics["candidate_cancelled"] += 1
                    self._cancel_background_task(candidate["task"])
                    metrics["candidate_cancel_overhead_ms"] += (time.perf_counter() - cancel_started_at) * 1000
                    self._candidate_fallback_ready_at = time.perf_counter()
                else:
                    candidate_output = await candidate["task"]
                    metrics["candidate_completed_before_validation"] += 1
                if candidate_output is not None:
                    elapsed_ms = (time.perf_counter() - candidate["started_at"]) * 1000
                    metrics["candidate_generation_ms"] += elapsed_ms
                    same_context = actual_context is not None and candidate_output.context_ids == actual_context
                    same_signature = (
                        candidate["policy_version"] == getattr(self.model, "current_policy_version", None)
                        and candidate["sampling_signature"] == getattr(self.model, "sampling_signature", "")
                        and candidate["tool_schema_signature"] == self.model.tool_schema_signature
                    )
                    can_commit = (
                        predicted_matches
                        and same_context
                        and same_signature
                        and candidate["has_no_environment_side_effect"]
                        and session_active
                        and tools_succeeded
                    )
                else:
                    can_commit = False
                if can_commit:
                    metrics["observation_matches"] += 1
                    metrics["candidate_committed"] += 1
                    committed_tool_time_s = sum(tr.execution_time or 0.0 for tr in tool_results)
                    metrics["hidden_tool_time"] += committed_tool_time_s
                    metrics["candidate_saved_overlap_ms"] += min(committed_tool_time_s * 1000, elapsed_ms)
                    if waited_after_tool_ms > 0:
                        metrics["candidate_committed_after_wait"] += 1
                        metrics["candidate_wait_committed_ms"] += waited_after_tool_ms
                    else:
                        metrics["candidate_committed_without_wait"] += 1
                    candidate_metrics = candidate_output.rollout_cache.setdefault("metrics", {})
                    for metric_name in TOOLCALL_METRIC_DEFAULTS:
                        candidate_metrics[metric_name] = metrics[metric_name]
                    if "tool_calls" in metrics:
                        candidate_metrics["tool_calls"] = metrics["tool_calls"]
                    self.pending_generation = candidate_output
                elif candidate_output is not None:
                    if waited_after_tool_ms > 0:
                        metrics["candidate_wait_validation_failed_ms"] += waited_after_tool_ms
                    if not predicted_matches:
                        metrics["candidate_abort_observation_mismatch"] += 1
                    elif not same_context:
                        metrics["candidate_abort_context_mismatch"] += 1
                    elif not same_signature:
                        metrics["candidate_abort_signature_mismatch"] += 1
                    elif not tools_succeeded:
                        metrics["candidate_abort_tool_failure"] += 1
                    elif not session_active:
                        metrics["candidate_abort_inactive_session"] += 1
                    metrics["candidate_aborted"] += 1
                    metrics["context_validation_failures"] += 1
                    metrics["candidate_wasted_ms"] += elapsed_ms
                    metrics["wasted_candidate_tokens"] += candidate_output.generation_info.get("completion_tokens", 0)
                else:
                    if not wait_timed_out:
                        if not predicted_matches:
                            metrics["candidate_abort_observation_mismatch"] += 1
                        elif not tools_succeeded:
                            metrics["candidate_abort_tool_failure"] += 1
                        elif not session_active:
                            metrics["candidate_abort_inactive_session"] += 1
                    metrics["candidate_aborted"] += 1
                    metrics["context_validation_failures"] += 1
            except Exception:
                metrics["candidate_abort_exception"] += 1
                metrics["candidate_aborted"] += 1
                metrics["context_validation_failures"] += 1
                if not candidate["task"].done():
                    self._cancel_background_task(candidate["task"])
            finally:
                metrics["candidate_validation_ms"] += (time.perf_counter() - validation_started_at) * 1000

        self.messages.extend(tool_messages)
        if actual_ids is not None:
            self.rollout_cache = self.model.append_encoded_messages_to_rollout_cache(actual_ids, self.rollout_cache)
        else:
            self.rollout_cache = await self.model.append_messages_to_rollout_cache(tool_messages, self.rollout_cache)
        step_output.tool_results = tool_results

        # step 7: step-level exit_reason (precedence: terminal_dead >
        # timeout_budget_exhausted > finished > completed_with_tool_errors > completed)
        if terminal_dead:
            step_output.done = True
            step_output.exit_reason = "terminal_dead"
            return step_output
        if self.timeout_budget < 0:
            step_output.done = True
            step_output.exit_reason = "timeout_budget_exhausted"
            self.logger.info("Exit step: timeout budget exhausted.")
            return step_output
        if saw_finish:
            step_output.done = True
            step_output.exit_reason = "finished"
            return step_output
        if any(tr.status in ("timeout", "syntax_error") for tr in tool_results):
            step_output.done = False
            step_output.exit_reason = "completed_with_tool_errors"
            return step_output
        step_output.done = False
        step_output.exit_reason = "completed"
        return step_output

    @auto_await
    async def run(self):
        self.trajectory: list[StepOutput] = []

        self.logger.info("Inital Prompt:")
        for message in self.messages:
            self.logger.info(f"{message['role'].upper()} PROMPT:\n{message['content']}")

        rollout_cache = await self.model.prepare_rollout_cache(self.messages)
        self.rollout_cache: dict[str, str] = rollout_cache
        initialize_toolcall_metrics(self.rollout_cache.setdefault("metrics", {}))

        done = False
        step_idx = 1
        format_retries = 0
        execution_time = time.perf_counter()
        while not done:
            step_started_at = time.perf_counter()
            try:
                step_output = await self.step(step_idx=step_idx)
                step_output.elapsed_time_s = time.perf_counter() - step_started_at
                self.trajectory.append(step_output)
                # Formatting retries are attempts at the same logical turn.
                if step_output.exit_reason == "format_error":
                    if format_retries >= self.max_format_retries_per_turn:
                        self.logger.error(
                            f"Exit after {self.max_format_retries_per_turn} format retries at turn {step_idx}."
                        )
                        self.trajectory.append(
                            StepOutput(step_idx=step_idx, done=True, exit_reason="format_retry_exhausted")
                        )
                        break
                    format_retries += 1
                    self.logger.info(
                        f"Retrying logical turn {step_idx} after format error "
                        f"({format_retries}/{self.max_format_retries_per_turn})."
                    )
                    continue

                format_retries = 0
                done = step_output.done
                if done:
                    break
                if step_idx >= self.max_turns:
                    self.logger.error(f"Exit due to max step limit: {self.max_turns}")
                    step_output = StepOutput(step_idx=step_idx, exit_reason="max_step_limit")
                    self.trajectory.append(step_output)
                    break
                step_idx += 1
            except Exception as e:
                # this should not happen, if it happens, we should fix the code
                _msg = (
                    f"[step{step_idx}] unknown_error: {type(e).__name__}: {e} "
                    f"response_mask_len_before={len(self.rollout_cache.get('response_mask', []))} "
                    f"prompt_ids_len={len(self.rollout_cache.get('prompt_ids', []))}"
                )
                self.logger.opt(exception=True).critical("{}", _msg)
                step_output = StepOutput(
                    step_idx=step_idx,
                    exit_reason="unknown_error",
                    elapsed_time_s=time.perf_counter() - step_started_at,
                )
                self.trajectory.append(step_output)
                break

        execution_time = time.perf_counter() - execution_time
        finalize_toolcall_metrics(self.rollout_cache.setdefault("metrics", {}), self.observation_cache)
        result = {
            "trajectory": self.trajectory,
            "rollout_cache": self.rollout_cache,
            "execution_time": execution_time,
            "messages": self.messages,
        }
        return result
