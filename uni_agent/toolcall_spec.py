"""Correctness primitives for tool-call speculation.

The observation cache in this module is deliberately separate from suffix
decoding.  A cache hit may start a model-only candidate generation, but the
real tool is still executed and its encoded observation is validated before a
candidate can be committed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

TERMINAL_TOOLS = frozenset({"finish", "submit"})
DEFAULT_ELIGIBLE_TOOLS = frozenset({"bash", "execute_bash", "str_replace_editor", "view"})
TOOL_EQUIVALENCE_CLASSES = {
    "shell": frozenset({"bash", "execute_bash"}),
    "file_view": frozenset({"str_replace_editor", "view"}),
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, set | frozenset):
        return sorted(value)
    return repr(value)


def stable_signature(value: Any) -> str:
    """Return a stable SHA-256 signature for JSON-like configuration."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_arguments(arguments: Any) -> str:
    """Serialize tool arguments while preserving JSON value types."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Invalid structured calls are rejected by the parser before they
            # reach speculation.  Keeping the raw string here makes this
            # helper deterministic when used directly in diagnostics/tests.
            pass
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _arguments_mapping(arguments: Any) -> Mapping[str, Any] | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, Mapping) else None


SAFE_SHELL_PATH_RE = re.compile(r"^[A-Za-z0-9_.+/-]+$")


def _equivalent_shell_call(arguments: Mapping[str, Any]) -> tuple[str, Any]:
    if set(arguments) != {"command"} or not isinstance(arguments.get("command"), str):
        return "execute_bash", dict(arguments)
    try:
        tokens = shlex.split(arguments["command"], posix=True)
    except ValueError:
        return "execute_bash", dict(arguments)
    if len(tokens) == 2 and tokens[0] == "cat":
        path = tokens[1]
        if not path.startswith("-") and SAFE_SHELL_PATH_RE.fullmatch(path):
            return "read_file", {"path": path}
    if len(tokens) == 3 and tokens[:2] == ["cat", "--"]:
        path = tokens[2]
        if SAFE_SHELL_PATH_RE.fullmatch(path):
            return "read_file", {"path": path}
    return "execute_bash", dict(arguments)


def equivalent_tool_call(name: str, arguments: Any) -> tuple[str, Any]:
    """Return a conservative operation identity shared by equivalent tools.

    Equivalence only affects observation-cache lookup. The real tool still
    runs and its encoded observation/context must match before a candidate is
    committed.
    """

    mapped_arguments = _arguments_mapping(arguments)
    if name in TOOL_EQUIVALENCE_CLASSES["shell"] and mapped_arguments is not None:
        return _equivalent_shell_call(mapped_arguments)

    if name == "str_replace_editor" and mapped_arguments is not None:
        if mapped_arguments.get("command") != "view":
            return name, arguments
        ignored_editor_fields = {"file_text", "old_str", "new_str", "insert_line"}
        if any(mapped_arguments.get(field) is not None for field in ignored_editor_fields):
            return name, arguments
        allowed_fields = {"command", "path", "view_range", *ignored_editor_fields}
        if set(mapped_arguments) - allowed_fields:
            return name, arguments
        view_arguments = {"path": mapped_arguments.get("path")}
        if mapped_arguments.get("view_range") is not None:
            view_arguments["view_range"] = mapped_arguments["view_range"]
        return "view", view_arguments

    if name == "view" and mapped_arguments is not None:
        allowed_fields = {"command", "path", "view_range"}
        if set(mapped_arguments) - allowed_fields or mapped_arguments.get("command", "view") != "view":
            return name, arguments
        view_arguments = {"path": mapped_arguments.get("path")}
        if mapped_arguments.get("view_range") is not None:
            view_arguments["view_range"] = mapped_arguments["view_range"]
        return "view", view_arguments

    return name, arguments


def tool_schema_signature(tool_schemas: list[dict[str, Any]] | None) -> str:
    return stable_signature(tool_schemas or [])


@dataclass(frozen=True)
class ToolcallCacheKey:
    tool_name: str
    canonical_arguments: str
    tool_schema_signature: str


def exact_tool_call_signature(tool_call: Any) -> str:
    """Identify the concrete call before equivalence normalization."""

    if isinstance(tool_call, Mapping):
        function = tool_call.get("function", tool_call)
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        function = getattr(tool_call, "function", tool_call)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", {})
    return stable_signature({"tool_name": name, "arguments": canonical_arguments(arguments)})


def canonical_tool_call(tool_call: Any, schema_signature: str) -> ToolcallCacheKey:
    """Build a cache key from either a pydantic or OpenAI-shaped tool call.

    The tool call id is intentionally ignored.  It is request linkage, not
    part of the operation identity, and the current id is always used when a
    predicted tool message is constructed.
    """

    if isinstance(tool_call, Mapping):
        function = tool_call.get("function", tool_call)
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        function = getattr(tool_call, "function", tool_call)
        name = function.name
        arguments = getattr(function, "arguments", {})

    if not isinstance(name, str) or not name:
        raise ValueError("tool call must contain a non-empty function name")
    name, arguments = equivalent_tool_call(name, arguments)
    return ToolcallCacheKey(
        tool_name=name,
        canonical_arguments=canonical_arguments(arguments),
        tool_schema_signature=schema_signature,
    )


@dataclass
class CachedObservation:
    content: str
    status: str
    source_trajectory_id: str
    created_at: float
    token_count: int = 0
    source_tool_name: str | None = None
    source_call_signature: str | None = None


class ObservationCache:
    """Token- and entry-bounded LRU of successful tool observations."""

    def __init__(
        self,
        max_entries: int = 1024,
        max_tokens: int = 1_000_000,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if max_entries < 0 or max_tokens < 0:
            raise ValueError("observation cache limits must be non-negative")
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._token_counter = token_counter or self._estimate_tokens
        self._entries: OrderedDict[ToolcallCacheKey, CachedObservation] = OrderedDict()
        self._tokens = 0

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        return max(1, (len(content.encode("utf-8")) + 3) // 4)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def token_count(self) -> int:
        return self._tokens

    def contains(self, key: ToolcallCacheKey) -> bool:
        return key in self._entries

    def get(self, key: ToolcallCacheKey) -> CachedObservation | None:
        observation = self._entries.get(key)
        if observation is not None:
            self._entries.move_to_end(key)
        return observation

    def put(
        self,
        key: ToolcallCacheKey,
        content: str,
        *,
        status: str,
        source_trajectory_id: str,
        created_at: float | None = None,
        token_count: int | None = None,
        source_tool_name: str | None = None,
        source_call_signature: str | None = None,
    ) -> bool:
        """Store the latest successful observation for ``key``.

        Returns whether the value was retained.  Timeout, syntax-error and
        skipped observations never enter the cache.
        """

        if status != "ok" or self.max_entries == 0 or self.max_tokens == 0:
            return False
        count = self._token_counter(content) if token_count is None else token_count
        if count < 0:
            raise ValueError("token_count must be non-negative")
        if count > self.max_tokens:
            return False

        old = self._entries.pop(key, None)
        if old is not None:
            self._tokens -= old.token_count
        observation = CachedObservation(
            content=content,
            status=status,
            source_trajectory_id=source_trajectory_id,
            created_at=time.time() if created_at is None else created_at,
            token_count=count,
            source_tool_name=source_tool_name,
            source_call_signature=source_call_signature,
        )
        self._entries[key] = observation
        self._tokens += count
        self._evict()
        return key in self._entries

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries or self._tokens > self.max_tokens:
            _, observation = self._entries.popitem(last=False)
            self._tokens -= observation.token_count

    def clear(self) -> None:
        self._entries.clear()
        self._tokens = 0


@dataclass
class CandidateGeneration:
    candidate_id: str
    cache_key: ToolcallCacheKey
    predicted_tool_message_ids: list[int]
    context_ids: list[int]
    policy_version: int | str | None
    sampling_signature: str
    tool_schema_signature: str
    task: asyncio.Task
    started_at: float
    has_no_environment_side_effect: bool = True


TOOLCALL_METRIC_DEFAULTS: dict[str, int | float] = {
    "toolcall_lookup_attempts": 0,
    "canonical_call_matches": 0,
    "equivalent_tool_matches": 0,
    "observation_matches": 0,
    "candidate_started": 0,
    "candidate_committed": 0,
    "candidate_aborted": 0,
    "wasted_candidate_tokens": 0,
    "candidate_generation_ms": 0.0,
    "candidate_wasted_ms": 0.0,
    "candidate_tool_execution_ms": 0.0,
    "candidate_validation_ms": 0.0,
    "candidate_wait_after_tool_ms": 0.0,
    "candidate_wait_committed_ms": 0.0,
    "candidate_wait_timeout_ms": 0.0,
    "candidate_wait_validation_failed_ms": 0.0,
    "candidate_committed_without_wait": 0,
    "candidate_committed_after_wait": 0,
    "candidate_abort_observation_mismatch": 0,
    "candidate_abort_wait_timeout": 0,
    "candidate_abort_context_mismatch": 0,
    "candidate_abort_signature_mismatch": 0,
    "candidate_abort_tool_failure": 0,
    "candidate_abort_inactive_session": 0,
    "candidate_abort_exception": 0,
    "candidate_saved_overlap_ms": 0.0,
    "candidate_cancel_overhead_ms": 0.0,
    "candidate_cancelled_elapsed_ms": 0.0,
    "candidate_fallback_start_delay_ms": 0.0,
    "candidate_cancel_requested": 0,
    "candidate_cancelled": 0,
    "candidate_completed_before_validation": 0,
    "hidden_tool_time": 0.0,
    "context_validation_failures": 0,
}


def initialize_toolcall_metrics(metrics: dict[str, Any]) -> None:
    for name, value in TOOLCALL_METRIC_DEFAULTS.items():
        metrics.setdefault(name, value)


def finalize_toolcall_metrics(metrics: dict[str, Any], cache: ObservationCache | None = None) -> None:
    """Add derived rates and cache gauges to a rollout metrics mapping."""

    attempts = int(metrics.get("toolcall_lookup_attempts", 0) or 0)
    canonical_matches = int(metrics.get("canonical_call_matches", 0) or 0)
    equivalent_matches = int(metrics.get("equivalent_tool_matches", 0) or 0)
    started = int(metrics.get("candidate_started", 0) or 0)
    committed = int(metrics.get("candidate_committed", 0) or 0)
    observation_matches = int(metrics.get("observation_matches", 0) or 0)
    metrics["canonical_call_match_rate"] = canonical_matches / attempts if attempts else 0.0
    metrics["equivalent_tool_match_rate"] = equivalent_matches / canonical_matches if canonical_matches else 0.0
    metrics["observation_match_rate"] = observation_matches / canonical_matches if canonical_matches else 0.0
    metrics["committed_candidate_rate"] = committed / started if started else 0.0
    if cache is not None:
        metrics["toolcall_cache_entries"] = len(cache)
        metrics["toolcall_cache_tokens"] = cache.token_count


def is_eligible_tool_call(
    *,
    enabled: bool,
    tool_calls: list[Any],
    cache: ObservationCache,
    cache_key: ToolcallCacheKey | None,
    allowed_tools: frozenset[str] = DEFAULT_ELIGIBLE_TOOLS,
) -> bool:
    """Apply the V1 single-call/read-only eligibility gate."""

    if not (enabled and len(tool_calls) == 1 and cache_key is not None):
        return False

    tool_call = tool_calls[0]
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function", tool_call)
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        function = getattr(tool_call, "function", tool_call)
        tool_name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", {})

    if tool_name in TERMINAL_TOOLS or tool_name not in allowed_tools or not cache.contains(cache_key):
        return False
    if tool_name == "str_replace_editor":
        mapped_arguments = _arguments_mapping(arguments)
        return mapped_arguments is not None and mapped_arguments.get("command") == "view"
    return True


__all__ = [
    "CachedObservation",
    "CandidateGeneration",
    "DEFAULT_ELIGIBLE_TOOLS",
    "ObservationCache",
    "TERMINAL_TOOLS",
    "TOOL_EQUIVALENCE_CLASSES",
    "ToolcallCacheKey",
    "canonical_arguments",
    "canonical_tool_call",
    "equivalent_tool_call",
    "exact_tool_call_signature",
    "finalize_toolcall_metrics",
    "initialize_toolcall_metrics",
    "is_eligible_tool_call",
    "stable_signature",
    "tool_schema_signature",
]
