"""Integration test for vLLM CPU-layer KV events."""

from __future__ import annotations

import time

import httpx
import pytest
from conftest import (
    CPU_KV_NODE_ID,
    CPU_KV_ZMQ_REPLAY_PORT,
    CPU_KV_ZMQ_SUB_PORT,
    VLLM_MODEL,
    send_inference_request,
)

from uni_agent.llm_router.collectors.collector import get_collector
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.store.data_store import DataStore
from uni_agent.llm_router.types import Layer


def _get_token_ids(node_id: str, model: str, prompt: str) -> list[int]:
    """Get prompt token IDs from vLLM's /tokenize endpoint."""
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
            return resp.json().get("tokens", [])
    except Exception:
        pass

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def _make_cpu_kv_collector():
    cfg = CollectorConfig()
    return get_collector(
        "vllm_zmq",
        cfg,
        kv_event_endpoints={
            CPU_KV_NODE_ID: [
                f"127.0.0.1:{CPU_KV_ZMQ_SUB_PORT}",
                f"127.0.0.1:{CPU_KV_ZMQ_REPLAY_PORT}",
                "zmq",
                "kv-events",
            ],
        },
    )


@pytest.mark.st
@pytest.mark.gpu
class TestVLLMCPUKVEventCollector:
    """Verify real vLLM CPU events update the CPU layer index."""

    def test_cpu_events_update_cpu_prefix_hit_rate(self, vllm_cpu_kv_service):
        prompt = " ".join(
            [
                "The CPU KV event integration test repeats a long deterministic prompt",
                "so that vLLM stores multiple prefix cache blocks and emits CPU medium KV events",
            ]
            * 8
        )
        prompt_ids = _get_token_ids(vllm_cpu_kv_service, VLLM_MODEL, prompt)
        assert len(prompt_ids) > 32, f"Expected a multi-block prompt, got {len(prompt_ids)} tokens"

        store = DataStore()
        collector = _make_cpu_kv_collector()
        collector.start()
        try:
            time.sleep(5.0)
            for _ in range(3):
                assert send_inference_request(vllm_cpu_kv_service, VLLM_MODEL, prompt)
                time.sleep(5.0)

            cpu_hit = 0.0
            deadline = time.time() + 45.0
            while time.time() < deadline:
                cpu_hit = store.get_layer_prefix_hit_rate(CPU_KV_NODE_ID, prompt_ids, Layer.CPU)
                if cpu_hit > 0.0:
                    break
                time.sleep(3.0)
        finally:
            collector.stop()

        assert cpu_hit > 0.0, f"Expected positive CPU layer prefix hit rate, got {cpu_hit}"
