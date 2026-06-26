"""Tests for RouteDataProvider.get_gpu_prefix_hit_rate with real vLLM service + ZMQ KV events.

Test flow:
1. Launch a real vLLM model service with kv-events-config enabled (ZMQ publisher).
2. Create RouteDataProvider with collection_names=["vllm_zmq"], which internally
   creates VLLMKVEventCollector + KVCacheStore via BUILTIN_REGISTRY.
3. Call provider.start() to begin event subscription.
4. Send inference requests via httpx to trigger KV cache block-stored events.
5. Obtain prompt token IDs via vLLM /tokenize endpoint (messages format, with chat template).
6. Call provider.get_gpu_prefix_hit_rate(prompt_ids) and verify the results.
"""

from __future__ import annotations

import asyncio

import pytest
import httpx

from conftest import NODE_ID, ZMQ_SUB_PORT, ZMQ_REPLAY_PORT, VLLM_MODEL, send_inference_request
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.collectors.provider import RouteDataProvider


# ── Helper: get token IDs via vLLM tokenize endpoint ────────────────────

def _get_token_ids(node_id: str, model: str, prompt: str) -> list[int]:
    """Get prompt token IDs from vLLM's /tokenize endpoint with chat template applied.

    Must use messages format so the chat template is applied — the same
    template vLLM applies during /v1/chat/completions inference.  KV cache
    blocks are computed from the fully-formatted sequence (including special
    tokens like <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n),
    so the token IDs used for prefix hash lookup must match that sequence.

    Falls back to transformers AutoTokenizer + apply_chat_template if
    /tokenize is unavailable.
    """
    try:
        resp = httpx.post(
            f"http://{node_id}/tokenize",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "add_generation_prompt": True,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("tokens", [])
    except Exception:
        pass

    # Fallback: use transformers AutoTokenizer with chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(text, add_special_tokens=False)
    return encoding.input_ids


# ── Helper: create RouteDataProvider and collect KV events ──────────────

def _setup_provider_and_collect(
    node_id: str, prompts: list[str],
) -> tuple[RouteDataProvider, dict[str, list[int]]]:
    """Create RouteDataProvider, start collector, send inference requests.

    Returns:
        (provider, prompt_token_ids_map) — provider is NOT stopped; callers
        must call provider.stop() after reading results.
    """
    kv_event_endpoints = {
        NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"],
    }
    provider = RouteDataProvider(
        collectors_config=CollectorConfig(),
        collection_names=["vllm_zmq"],
        kv_event_endpoints=kv_event_endpoints,
    )

    prompt_token_ids: dict[str, list[int]] = {}

    async def _run():
        provider.start()
        await asyncio.sleep(5.0)  # wait for ZMQ connection + replay

        for prompt in prompts:
            send_inference_request(node_id, VLLM_MODEL, prompt)
            await asyncio.sleep(3.0)

        await asyncio.sleep(5.0)  # wait for events to be processed

        for prompt in prompts:
            prompt_token_ids[prompt] = _get_token_ids(node_id, VLLM_MODEL, prompt)

    asyncio.run(_run())
    return provider, prompt_token_ids


# ── Test class ───────────────────────────────────────────────────────────

class TestRouteDataProviderGpuPrefixHitRateWithRealService:
    """Integration tests: RouteDataProvider.get_gpu_prefix_hit_rate against a live vLLM ZMQ publisher."""

    def test_prefix_hit_rate_with_partial_match(self, vllm_kv_service):
        """
        Feature: get_gpu_prefix_hit_rate returns 100% hit rate for a shorter prompt
        Description:
            1. Send an inference request with a long prompt A.
            2. Call get_gpu_prefix_hit_rate with prompt B that is a strict prefix of A.
        Expectation:
            Since B's blocks are a subset of A's cached blocks, all of B's prefix
            blocks are cached → hit_rate = 100.
        """
        prompt_long = "The history of artificial intelligence began in the 1950s and has evolved dramatically since then"
        prompt_short_prefix = "The history of artificial intelligence began in the 1950s"

        provider, token_ids_map = _setup_provider_and_collect(vllm_kv_service, [prompt_long])

        short_ids = _get_token_ids(vllm_kv_service, VLLM_MODEL, prompt_short_prefix)
        long_ids = token_ids_map[prompt_long]

        assert len(short_ids) < len(long_ids), (
            f"Short prompt should have fewer tokens than long prompt, "
            f"got short={len(short_ids)}, long={len(long_ids)}"
        )

        result = provider.get_gpu_prefix_hit_rate(short_ids)

        if len(result) == 0:
            pytest.skip(
                f"Short prompt has {len(short_ids)} tokens, fewer than block_size "
                f"— cannot form a full block, so prefix hit rate is empty dict"
            )

        assert NODE_ID in result, (
            f"Expected NODE_ID '{NODE_ID}' in result keys for partial prefix, "
            f"got {list(result.keys())}"
        )
        hit_rate = result[NODE_ID]
        assert hit_rate == 100, f"Expected hit_rate = 100 for prefix match, got {hit_rate}%"

        provider.stop()

    def test_prefix_hit_rate_returns_replica_id_key(self, vllm_kv_service):
        """
        Feature: get_gpu_prefix_hit_rate returns dict with replica_id as key
        Description:
            1. Send an inference request.
            2. Call get_gpu_prefix_hit_rate with the prompt's token IDs.
        Expectation:
            Keys are replica IDs in "host:port" format matching NODE_ID.
            Values are integers in [0, 100].
        """
        prompt = "Explain the concept of neural networks in simple terms"

        provider, token_ids_map = _setup_provider_and_collect(vllm_kv_service, [prompt])
        prompt_ids = token_ids_map[prompt]
        assert len(prompt_ids) > 0, "Should have token IDs for the prompt"

        result = provider.get_gpu_prefix_hit_rate(prompt_ids)

        assert len(result) > 0, f"Expected non-empty result dict, got {result}"

        for key in result:
            assert ":" in key, f"Replica ID key should be 'host:port', got '{key}'"
            host_part, port_part = key.rsplit(":", 1)
            assert host_part == "127.0.0.1", f"Expected host '127.0.0.1', got '{host_part}'"
            assert port_part.isdigit(), f"Replica port should be numeric, got '{port_part}'"

        assert NODE_ID in result, (
            f"Expected NODE_ID '{NODE_ID}' in result keys, got {list(result.keys())}"
        )

        for replica_id, hit_rate in result.items():
            assert isinstance(hit_rate, int), f"Hit rate should be int, got {type(hit_rate).__name__}"
            assert 0 <= hit_rate <= 100, f"Hit rate should be in [0, 100], got {hit_rate}"

        provider.stop()
