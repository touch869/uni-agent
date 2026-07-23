"""SPEC-RL draft reuse for Uni-Agent rollout clients.

The implementation operates on one model call at a time.  It never reuses tool
outputs or an entire agent trajectory, so environment interactions still run in
the current session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import ray
import torch

from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)

_SPEC_COUNTERS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
    "spec_cache_hits",
    "spec_cache_misses",
    "spec_fallbacks",
    "spec_saved_tokens",
    "spec_version_older_bypass",
    "spec_version_equal_bypass",
    "spec_version_newer_verify",
    "spec_verify_prompt_tokens",
    "spec_verify_draft_tokens",
    "spec_continuation_tokens",
    "spec_continuation_cached_tokens",
    "spec_continuation_prefill_tokens",
    "spec_fallback_unsupported",
    "spec_fallback_cache_miss",
    "spec_fallback_older_policy",
    "spec_fallback_equal_policy",
    "spec_fallback_verification_error",
    "spec_fallback_missing_logprobs",
    "spec_fallback_continuation_error",
)

_SPEC_TIMINGS = (
    "spec_cache_lookup_ms",
    "spec_version_check_ms",
    "spec_verify_ms",
    "spec_continuation_ms",
    "spec_normal_fallback_ms",
)


@dataclass(frozen=True)
class SpecRLSettings:
    enabled: bool = False
    bias: float = 0.5
    seed: int = 1234
    cache_max_entries: int = 10_000
    cache_max_tokens: int = 10_000_000
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0

    @classmethod
    def from_mapping(cls, value: Any, *, rollout_config: Any) -> SpecRLSettings:
        value = value or {}
        get = value.get if hasattr(value, "get") else lambda key, default=None: default
        return cls(
            enabled=bool(get("enabled", False)),
            bias=float(get("bias", 0.5)),
            seed=int(get("seed", 1234)),
            cache_max_entries=int(get("cache_max_entries", 10_000)),
            cache_max_tokens=int(get("cache_max_tokens", 10_000_000)),
            temperature=float(rollout_config.get("temperature", 1.0)),
            top_p=float(rollout_config.get("top_p", 1.0)),
            top_k=int(rollout_config.get("top_k", -1)),
            repetition_penalty=float(rollout_config.get("repetition_penalty", 1.0)),
        )

    def validate(self) -> None:
        if self.bias < 0:
            raise ValueError(f"SPEC-RL bias must be non-negative, got {self.bias}")
        if self.cache_max_entries <= 0 or self.cache_max_tokens <= 0:
            raise ValueError("SPEC-RL cache limits must be positive")


def accepted_prefix_length(
    old_logprobs: list[float],
    new_logprobs: list[float],
    *,
    bias: float,
    seed: int,
) -> int:
    """Return the longest prefix accepted by the static SPEC-RL rule."""
    if len(old_logprobs) != len(new_logprobs):
        raise ValueError("old and new log probabilities must have identical lengths")
    if bias < 0:
        raise ValueError("bias must be non-negative")

    rng = random.Random(seed)
    for index, (old_logp, new_logp) in enumerate(zip(old_logprobs, new_logprobs, strict=True)):
        if not math.isfinite(old_logp) or not math.isfinite(new_logp):
            return index
        log_u = math.log(max(rng.random(), 1e-12))
        if log_u > new_logp - old_logp + bias:
            return index
    return len(old_logprobs)


class SpecRLDraftCache:
    """Token-bounded LRU cache hosted by one shared Ray actor."""

    def __init__(self, max_entries: int, max_tokens: int):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._token_count = 0

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(
        self,
        key: str,
        token_ids: list[int],
        logprobs: list[float],
        stop_reason: str | None,
        policy_version: int,
    ) -> bool:
        if not token_ids or len(token_ids) != len(logprobs) or len(token_ids) > self.max_tokens:
            return False

        existing = self._entries.get(key)
        if existing is not None and int(existing["policy_version"]) > policy_version:
            return False
        if existing is not None:
            self._token_count -= int(existing["token_ids"].numel())
            del self._entries[key]

        entry = {
            "token_ids": torch.tensor(token_ids, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "stop_reason": stop_reason,
            "policy_version": int(policy_version),
        }
        self._entries[key] = entry
        self._token_count += len(token_ids)
        while len(self._entries) > self.max_entries or self._token_count > self.max_tokens:
            _, evicted = self._entries.popitem(last=False)
            self._token_count -= int(evicted["token_ids"].numel())
        return True

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "tokens": self._token_count}


SpecRLDraftCacheActor = ray.remote(SpecRLDraftCache)


def _counter_fields(**updates: int | float) -> dict[str, int | float]:
    counters: dict[str, int | float] = {key: 0 for key in _SPEC_COUNTERS}
    counters.update({key: 0.0 for key in _SPEC_TIMINGS})
    counters.update(updates)
    return counters


def _merge_extra_fields(output: TokenOutput, counters: dict[str, int | float], **extra: Any) -> TokenOutput:
    for key in _SPEC_COUNTERS:
        output.extra_fields[key] = int(output.extra_fields.get(key, 0)) + int(counters.get(key, 0))
    for key in _SPEC_TIMINGS:
        output.extra_fields[key] = float(output.extra_fields.get(key, 0.0)) + float(counters.get(key, 0.0))
    output.extra_fields.update(extra)
    return output


class SpecRLLLMServerClient:
    """Decorate an LLM client with history-based speculative rollouts."""

    def __init__(self, base_client: Any, cache: Any, settings: SpecRLSettings):
        settings.validate()
        self.base_client = base_client
        self.cache = cache
        self.settings = settings

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        specrl_enabled: bool = False,
        **kwargs: Any,
    ) -> TokenOutput:
        call_kwargs = {
            "request_id": request_id,
            "prompt_ids": prompt_ids,
            "sampling_params": sampling_params,
            "image_data": image_data,
            "video_data": video_data,
            "audio_data": audio_data,
            "mm_processor_kwargs": mm_processor_kwargs,
            **kwargs,
        }
        if not specrl_enabled or not self.settings.enabled:
            return await self.base_client.generate(**call_kwargs)

        unsupported_reason = self._unsupported_reason(
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )
        if unsupported_reason is not None:
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_fallbacks=1,
                    spec_fallback_unsupported=1,
                ),
                reason="unsupported",
                with_logprobs=False,
            )

        key = self._cache_key(prompt_ids, sampling_params)
        lookup_started = time.perf_counter()
        try:
            cached = await self._cache_call("get", key)
        except Exception:
            logger.exception("SPEC-RL cache lookup failed for request %s", request_id)
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_fallbacks=1,
                    spec_fallback_verification_error=1,
                    spec_cache_lookup_ms=(time.perf_counter() - lookup_started) * 1000.0,
                ),
                reason="verification_error",
            )
        lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
        if cached is None:
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_cache_misses=1,
                    spec_fallback_cache_miss=1,
                    spec_cache_lookup_ms=lookup_ms,
                ),
                reason="cache_miss",
                cache_key=key,
                count_as_fallback=False,
            )
        return await self._generate_from_draft(key, cached, call_kwargs, lookup_ms=lookup_ms)

    async def _generate_from_draft(
        self,
        key: str,
        cached: dict[str, Any],
        call_kwargs: dict[str, Any],
        *,
        lookup_ms: float,
    ) -> TokenOutput:
        draft_ids = self._as_list(cached["token_ids"], int)
        old_logprobs = self._as_list(cached["logprobs"], float)
        cached_version = int(cached["policy_version"])
        request_id = str(call_kwargs["request_id"])

        version_started = time.perf_counter()
        try:
            current_version = int(await self.base_client.get_policy_version(request_id=request_id))
        except Exception:
            logger.exception("SPEC-RL policy-version check failed for request %s", request_id)
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_cache_hits=1,
                    spec_fallbacks=1,
                    spec_fallback_verification_error=1,
                    spec_cache_lookup_ms=lookup_ms,
                    spec_version_check_ms=(time.perf_counter() - version_started) * 1000.0,
                ),
                reason="verification_error",
                cache_key=key,
            )
        version_ms = (time.perf_counter() - version_started) * 1000.0

        if current_version <= cached_version:
            older = current_version < cached_version
            reason = "older_policy" if older else "equal_policy"
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_cache_hits=1,
                    spec_fallbacks=1,
                    spec_version_older_bypass=int(older),
                    spec_version_equal_bypass=int(not older),
                    spec_fallback_older_policy=int(older),
                    spec_fallback_equal_policy=int(not older),
                    spec_cache_lookup_ms=lookup_ms,
                    spec_version_check_ms=version_ms,
                ),
                reason=reason,
                cache_key=key,
            )

        seed_material = f"{self.settings.seed}:{key}:{current_version}".encode()
        accept_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        try:
            output = await self.base_client.verify_and_generate(
                request_id=request_id,
                prompt_ids=list(call_kwargs["prompt_ids"]),
                draft_ids=draft_ids,
                old_logprobs=old_logprobs,
                sampling_params=dict(call_kwargs["sampling_params"]),
                bias=self.settings.bias,
                accept_seed=accept_seed,
                expected_policy_version=current_version,
                cached_stop_reason=cached["stop_reason"],
                priority=call_kwargs.get("priority", 0),
            )
        except Exception:
            logger.exception("SPEC-RL verify-and-generate failed for request %s", request_id)
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(
                    spec_cache_hits=1,
                    spec_fallbacks=1,
                    spec_version_newer_verify=1,
                    spec_fallback_verification_error=1,
                    spec_cache_lookup_ms=lookup_ms,
                    spec_version_check_ms=version_ms,
                ),
                reason="verification_error",
                cache_key=key,
            )

        error_reason = output.extra_fields.pop("_specrl_error_reason", None)
        if error_reason is not None:
            reason_field = {
                "missing_logprobs": "spec_fallback_missing_logprobs",
                "continuation_error": "spec_fallback_continuation_error",
            }.get(str(error_reason), "spec_fallback_verification_error")
            operation_metrics = {name: output.extra_fields.get(name, 0) for name in (*_SPEC_COUNTERS, *_SPEC_TIMINGS)}
            operation_metrics.update(
                {
                    "spec_cache_hits": 1,
                    "spec_fallbacks": 1,
                    "spec_version_newer_verify": 1,
                    "spec_cache_lookup_ms": lookup_ms,
                    "spec_version_check_ms": version_ms,
                    reason_field: 1,
                }
            )
            return await self._normal_fallback(
                call_kwargs,
                _counter_fields(**operation_metrics),
                reason=str(error_reason),
                cache_key=key,
            )

        await self._store_output(key, output)
        return _merge_extra_fields(
            output,
            _counter_fields(
                spec_cache_hits=1,
                spec_version_newer_verify=1,
                spec_cache_lookup_ms=lookup_ms,
                spec_version_check_ms=version_ms,
            ),
        )

    async def _normal_fallback(
        self,
        call_kwargs: dict[str, Any],
        counters: dict[str, int | float],
        *,
        reason: str,
        cache_key: str | None = None,
        with_logprobs: bool = True,
        count_as_fallback: bool = True,
    ) -> TokenOutput:
        started = time.perf_counter()
        if with_logprobs:
            output = await self._generate_with_logprobs(call_kwargs)
        else:
            output = await self.base_client.generate(**call_kwargs)
        counters["spec_normal_fallback_ms"] = (
            float(counters.get("spec_normal_fallback_ms", 0.0)) + (time.perf_counter() - started) * 1000.0
        )
        if cache_key is not None and with_logprobs:
            await self._store_output(cache_key, output)
        if not count_as_fallback:
            counters["spec_fallbacks"] = 0
        return _merge_extra_fields(output, counters, spec_fallback_reason=reason)

    async def _generate_with_logprobs(self, call_kwargs: dict[str, Any]) -> TokenOutput:
        request = dict(call_kwargs)
        request["sampling_params"] = dict(call_kwargs["sampling_params"])
        request["sampling_params"]["logprobs"] = True
        output = await self.base_client.generate(**request)
        if output.log_probs is None:
            raise ValueError("rollout server did not return response log probabilities")
        return output

    async def _store_output(self, key: str, output: TokenOutput) -> None:
        if output.log_probs is None or not output.token_ids:
            return
        await self._cache_call(
            "put",
            key,
            list(output.token_ids),
            list(output.log_probs),
            output.stop_reason,
            self._policy_version(output),
        )

    async def _cache_call(self, method: str, *args: Any) -> Any:
        fn = getattr(self.cache, method)
        if hasattr(fn, "remote"):
            return await fn.remote(*args)
        result = fn(*args)
        if hasattr(result, "__await__"):
            return await result
        return result

    def _unsupported_reason(
        self,
        *,
        sampling_params: dict[str, Any],
        image_data: list[Any] | None,
        video_data: list[Any] | None,
        audio_data: list[Any] | None,
    ) -> str | None:
        if image_data or video_data or audio_data:
            return "multimodal"
        temperature = float(sampling_params.get("temperature", self.settings.temperature))
        top_p = float(sampling_params.get("top_p", self.settings.top_p))
        top_k = int(sampling_params.get("top_k", self.settings.top_k))
        repetition_penalty = float(sampling_params.get("repetition_penalty", self.settings.repetition_penalty))
        presence_penalty = float(sampling_params.get("presence_penalty", 0.0))
        frequency_penalty = float(sampling_params.get("frequency_penalty", 0.0))
        min_p = float(sampling_params.get("min_p", 0.0))
        if (
            temperature != 1.0
            or top_p != 1.0
            or top_k != -1
            or repetition_penalty != 1.0
            or presence_penalty != 0.0
            or frequency_penalty != 0.0
            or min_p != 0.0
        ):
            return "unsupported_sampling"
        max_tokens = sampling_params.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return "missing_max_tokens"
        return None

    @staticmethod
    def _policy_version(output: TokenOutput) -> int:
        value = output.extra_fields.get("max_global_steps", output.extra_fields.get("global_steps"))
        if value is None:
            raise ValueError("rollout output is missing its model policy version")
        return int(value)

    @staticmethod
    def _as_list(value: Any, cast: type) -> list[Any]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [cast(item) for item in value]

    @staticmethod
    def _merge_policy_versions(target: dict[str, Any], scored: dict[str, Any]) -> None:
        versions = [
            value
            for value in (
                target.get("min_global_steps"),
                target.get("global_steps"),
                scored.get("min_global_steps"),
                scored.get("global_steps"),
            )
            if value is not None
        ]
        max_versions = [
            value
            for value in (
                target.get("max_global_steps"),
                target.get("global_steps"),
                scored.get("max_global_steps"),
                scored.get("global_steps"),
            )
            if value is not None
        ]
        if versions:
            target["min_global_steps"] = min(int(value) for value in versions)
        if max_versions:
            target["max_global_steps"] = max(int(value) for value in max_versions)

    @staticmethod
    def _cache_key(prompt_ids: list[int], sampling_params: dict[str, Any]) -> str:
        fingerprint_keys = (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "ignore_eos",
            "stop",
            "stop_token_ids",
        )
        fingerprint = {key: sampling_params.get(key) for key in fingerprint_keys}
        digest = hashlib.sha256(array("q", prompt_ids).tobytes())
        digest.update(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()


def spec_timing_names() -> tuple[str, ...]:
    return _SPEC_TIMINGS


def spec_counter_names() -> tuple[str, ...]:
    return _SPEC_COUNTERS
