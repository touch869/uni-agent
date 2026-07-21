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
    def from_mapping(cls, value: Any, *, rollout_config: Any) -> "SpecRLSettings":
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


def _counter_fields(**updates: int) -> dict[str, int]:
    counters = {key: 0 for key in _SPEC_COUNTERS}
    counters.update(updates)
    return counters


def _merge_extra_fields(output: TokenOutput, counters: dict[str, int], **extra: Any) -> TokenOutput:
    output.extra_fields.update(counters)
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

        fallback_reason = self._unsupported_reason(
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )
        if fallback_reason is not None:
            output = await self.base_client.generate(**call_kwargs)
            return _merge_extra_fields(
                output,
                _counter_fields(spec_fallbacks=1),
                spec_fallback_reason=fallback_reason,
            )

        key = self._cache_key(prompt_ids, sampling_params)
        try:
            cached = await self._cache_call("get", key)
            if cached is None:
                output = await self._generate_with_logprobs(call_kwargs)
                await self._store_output(key, output)
                return _merge_extra_fields(output, _counter_fields(spec_cache_misses=1))
            return await self._generate_from_draft(key, cached, call_kwargs)
        except Exception as exc:
            logger.warning("SPEC-RL fallback for request %s: %s: %s", request_id, type(exc).__name__, exc)
            output = await self.base_client.generate(**call_kwargs)
            return _merge_extra_fields(
                output,
                _counter_fields(spec_fallbacks=1),
                spec_fallback_reason=type(exc).__name__,
            )

    async def _generate_from_draft(
        self,
        key: str,
        cached: dict[str, Any],
        call_kwargs: dict[str, Any],
    ) -> TokenOutput:
        draft_ids = self._as_list(cached["token_ids"], int)
        old_logprobs = self._as_list(cached["logprobs"], float)
        prompt_ids = list(call_kwargs["prompt_ids"])

        score_output = await self.base_client.generate(
            request_id=call_kwargs["request_id"],
            prompt_ids=prompt_ids + draft_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "prompt_logprobs": 0,
                "logprobs": False,
            },
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
        )
        current_version = self._policy_version(score_output)
        cached_version = int(cached["policy_version"])
        if current_version <= cached_version:
            output = await self._generate_with_logprobs(call_kwargs)
            await self._store_output(key, output)
            return _merge_extra_fields(
                output,
                _counter_fields(spec_cache_hits=1, spec_fallbacks=1),
                spec_fallback_reason="non_newer_policy",
            )

        prompt_logprobs = score_output.extra_fields.get("prompt_logprobs")
        if prompt_logprobs is None:
            raise ValueError("rollout server did not return prompt_logprobs")
        start = len(prompt_ids)
        new_logprobs = [float(row[0]) for row in prompt_logprobs[start : start + len(draft_ids)]]
        if len(new_logprobs) != len(draft_ids):
            raise ValueError("prompt_logprobs length does not cover the cached draft")

        seed_material = f"{self.settings.seed}:{key}:{current_version}".encode()
        accept_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        accepted = accepted_prefix_length(
            old_logprobs,
            new_logprobs,
            bias=self.settings.bias,
            seed=accept_seed,
        )
        counters = _counter_fields(
            spec_num_draft_tokens=len(draft_ids),
            spec_num_accepted_tokens=accepted,
            spec_num_verify_steps=1,
            spec_cache_hits=1,
            spec_saved_tokens=accepted,
        )

        max_tokens = int(call_kwargs["sampling_params"]["max_tokens"])
        if accepted == len(draft_ids) or accepted >= max_tokens:
            score_extra_fields = {
                key: value
                for key, value in score_output.extra_fields.items()
                if key not in {"prompt_ids", "prompt_logprobs"}
            }
            output = TokenOutput(
                token_ids=draft_ids[:max_tokens],
                log_probs=new_logprobs[:max_tokens],
                stop_reason=cached["stop_reason"] if accepted == len(draft_ids) else "length",
                extra_fields=score_extra_fields,
            )
        else:
            continuation_kwargs = dict(call_kwargs)
            continuation_kwargs["prompt_ids"] = prompt_ids + draft_ids[:accepted]
            continuation_params = dict(call_kwargs["sampling_params"])
            continuation_params["max_tokens"] = max_tokens - accepted
            continuation_params["logprobs"] = True
            continuation_kwargs["sampling_params"] = continuation_params
            continuation = await self.base_client.generate(**continuation_kwargs)
            if continuation.log_probs is None:
                raise ValueError("continuation did not return log probabilities")
            output = TokenOutput(
                token_ids=draft_ids[:accepted] + list(continuation.token_ids),
                log_probs=new_logprobs[:accepted] + list(continuation.log_probs),
                routed_experts=continuation.routed_experts,
                stop_reason=continuation.stop_reason,
                num_preempted=continuation.num_preempted,
                extra_fields=dict(continuation.extra_fields),
            )
            self._merge_policy_versions(output.extra_fields, score_output.extra_fields)

        await self._store_output(key, output)
        return _merge_extra_fields(output, counters)

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


def spec_counter_names() -> tuple[str, ...]:
    return _SPEC_COUNTERS
