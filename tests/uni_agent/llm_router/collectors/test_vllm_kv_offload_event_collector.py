"""Integration test for native vLLM CPU KV offload events."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from uni_agent.llm_router.collectors.collector.vllm.event_collector import (
    VLLMKVEventCollector,
)
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore


VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_OFFLOAD_PORT = int(os.environ.get("VLLM_OFFLOAD_PORT", "8001"))
NODE_ID = f"{VLLM_HOST}:{VLLM_OFFLOAD_PORT}"

ZMQ_OFFLOAD_SUB_PORT = int(os.environ.get("ZMQ_OFFLOAD_SUB_PORT", "5565"))
ZMQ_OFFLOAD_REPLAY_PORT = int(os.environ.get("ZMQ_OFFLOAD_REPLAY_PORT", "5566"))
VLLM_CPU_OFFLOAD_GB = float(os.environ.get("VLLM_CPU_OFFLOAD_GB", "2"))
VLLM_OFFLOAD_PROMPT_WORDS = int(os.environ.get("VLLM_OFFLOAD_PROMPT_WORDS", "768"))


class _ZMQCollectorConfigMock:
    def __init__(
        self,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self.long_connection = {
            "base_retry_delay": base_retry_delay,
            "max_retry_delay": max_retry_delay,
            "max_retry_attempts": max_retry_attempts,
            "retry_backoff_factor": retry_backoff_factor,
        }


class _RecordingVLLMKVEventCollector(VLLMKVEventCollector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events_by_medium: dict[str, int] = {}

    def _apply_event(
        self,
        event: KVCacheEvent,
        default_replica_id: str | None = None,
    ) -> None:
        medium = (event.medium or "gpu").lower()
        self.events_by_medium[medium] = self.events_by_medium.get(medium, 0) + 1
        super()._apply_event(event, default_replica_id=default_replica_id)


@pytest.fixture(scope="module")
def vllm_kv_offload_service():
    """Launch vLLM with KV events and native CPU KV offload enabled."""
    kv_events_config = json.dumps({
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": f"tcp://*:{ZMQ_OFFLOAD_SUB_PORT}",
        "replay_endpoint": f"tcp://*:{ZMQ_OFFLOAD_REPLAY_PORT}",
    })

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        VLLM_MODEL,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_OFFLOAD_PORT),
        "--trust-remote-code",
        "--tensor_parallel_size",
        "2",
        "--dtype",
        "bfloat16",
        "--gpu_memory_utilization",
        "0.6",
        "--max-model-len",
        "8192",
        "--override_generation_config",
        '{"temperature": 0.8, "top_k": -1, "top_p": 0.9, "repetition_penalty": 1.0, "max_new_tokens": 4096}',
        "--kv-events-config",
        kv_events_config,
        "--kv-offloading-size",
        str(VLLM_CPU_OFFLOAD_GB),
        "--kv-offloading-backend",
        "native",
    ]

    proc = subprocess.Popen(cmd)

    metrics_url = f"http://{NODE_ID}/metrics"
    max_wait = 180
    deadline = time.time() + max_wait
    ready = False

    while time.time() < deadline:
        try:
            resp = httpx.get(metrics_url, timeout=5.0)
            if resp.status_code == 200:
                ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(3)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        pytest.skip(
            f"vLLM offload server for {VLLM_MODEL} did not become ready within {max_wait}s"
        )

    yield NODE_ID

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _long_repeated_prompt(num_words: int = VLLM_OFFLOAD_PROMPT_WORDS) -> str:
    """Build a deterministic long prompt that spans many KV blocks."""
    return " ".join("hello" for _ in range(num_words))


def _send_inference_request(node_id: str, model: str, prompt: str) -> tuple[bool, str]:
    try:
        resp = httpx.post(
            f"http://{node_id}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        detail = f"status={resp.status_code} body={resp.text[:500]}"
        return resp.status_code == 200, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def test_native_cpu_offload_events_update_cpu_tier(vllm_kv_offload_service):
    """Native vLLM CPU KV offload events update the CPU tier."""
    config = _ZMQCollectorConfigMock(base_retry_delay=1.0, max_retry_attempts=5)
    collector = _RecordingVLLMKVEventCollector(
        config,
        kv_event_addresses={
            NODE_ID: [
                f"127.0.0.1:{ZMQ_OFFLOAD_SUB_PORT}",
                f"127.0.0.1:{ZMQ_OFFLOAD_REPLAY_PORT}",
            ]
        },
    )
    store = KVCacheStore()
    prompt = _long_repeated_prompt()

    async def _run_subscribe():
        collector.start(store)
        await asyncio.sleep(5.0)

        for _ in range(2):
            ok, detail = _send_inference_request(
                vllm_kv_offload_service, VLLM_MODEL, prompt
            )
            if not ok:
                collector.stop()
                pytest.skip(
                    "Inference request failed; cannot validate CPU offload events: "
                    + detail
                )
            await asyncio.sleep(3.0)

        deadline = time.time() + 90
        while time.time() < deadline:
            has_cpu_entry = any(
                tier == "cpu" and NODE_ID in replicas
                for (tier, _), replicas in store.replicas_by_tier_and_block.items()
            )
            if has_cpu_entry:
                break
            await asyncio.sleep(2.0)

        collector.stop()

    asyncio.run(_run_subscribe())

    cpu_entries = {
        block_hash: replicas
        for (tier, block_hash), replicas in store.replicas_by_tier_and_block.items()
        if tier == "cpu"
    }
    if not cpu_entries:
        cpu_event_count = collector.events_by_medium.get("cpu", 0)
        if cpu_event_count > 0:
            pytest.fail(
                "Native CPU KV offload events were observed, but none were "
                "translated into CPU tier entries. This usually means remote "
                "hash mapping was missing for CPU BlockStored events."
            )
        pytest.skip(
            "No native CPU KV offload events observed; model/config may not "
            "trigger offload reliably"
        )

    assert any(NODE_ID in replicas for replicas in cpu_entries.values())
    assert NODE_ID in store.cpu_tracking_replicas
    assert store.get_replicas(next(iter(cpu_entries)), tier="cpu") is not None
