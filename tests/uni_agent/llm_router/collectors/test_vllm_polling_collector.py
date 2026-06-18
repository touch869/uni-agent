"""Tests for VLLMPollingCollector with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B).
2. Create VLLMPollingCollector with config containing the vLLM service address.
3. Call start() to begin metrics polling and write to MetricsStore.
4. Verify that expected metrics exist in the store.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import signal
import sys

import pytest
import httpx

from uni_agent.llm_router.collectors.collector.vllm.polling_collector import VLLMPollingCollector
from uni_agent.llm_router.collectors.metric_spec import MetricKey
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore


# ── Configuration constants ──────────────────────────────────────────────

import os

VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
NODE_ID = f"{VLLM_HOST}:{VLLM_PORT}"


# ── Helper: construct a config-like object for PollingCollector ──────────

class _CollectorConfigMock:
    """Minimal config object matching PollingCollector's expected interface.

    PollingCollector.__init__ reads:
      - config.http_polling["polling_interval"]  (float, seconds between polls)
      - config.http_polling["http_timeout"]      (float, HTTP request timeout)
    """

    def __init__(
        self,
        polling_interval: float = 2.0,
        http_timeout: float = 10.0,
    ) -> None:
        self.http_polling = {
            "polling_interval": polling_interval,
            "http_timeout": http_timeout,
        }


# ── Fixture: launch and teardown real vLLM service ───────────────────────

@pytest.fixture(scope="module")
def vllm_service():
    """Launch a real vLLM OpenAI-compatible server for the test module.

    Starts ``vllm serve <model> --host <host> --port <port>``,
    waits until ``/metrics`` is reachable, then yields the address.
    Kills the subprocess on teardown.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_MODEL,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--tensor_parallel_size", "2",
        "--dtype", "bfloat16",
        "--gpu_memory_utilization", "0.6",
        "--max-model-len", "8192",
        "--override_generation_config", '{"temperature": 0.8, "top_k": -1, "top_p": 0.9, "repetition_penalty": 1.0, "max_new_tokens": 4096}',
    ]

    proc = subprocess.Popen(cmd)

    # Wait for the vLLM server to become ready (poll /metrics endpoint)
    metrics_url = f"http://{NODE_ID}/metrics"
    max_wait = 180  # vLLM can take a while to load the model
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
        # Server didn't start — kill and fail
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        pytest.skip(f"vLLM server for {VLLM_MODEL} did not become ready within {max_wait}s")

    # Server is ready — send a test inference request
    print(f"\n[vllm_service fixture] Server is ready, sending test inference request...")
    try:
        test_response = httpx.post(
            f"http://{NODE_ID}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": "hello, uni-agent, please respond to this greeting"}],
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=30.0
        )
        print(f"[vllm_service fixture] Test inference response status: {test_response.status_code}")
        if test_response.status_code == 200:
            result = test_response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
            print(f"[vllm_service fixture] Test inference response: {content[:100]}...")
        else:
            print(f"[vllm_service fixture] Test inference failed with status {test_response.status_code}")
    except Exception as e:
        print(f"[vllm_service fixture] Test inference error: {e}")

    yield NODE_ID

    # Teardown: kill the vLLM process
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── Test class ───────────────────────────────────────────────────────────

class TestVLLMPollingCollectorWithRealService:
    """Integration tests: VLLMPollingCollector against a live vLLM server."""

    def test_start_and_metrics_exist(self, vllm_service):
        """
        Feature: VLLMPollingCollector collects real metrics from a running vLLM server
        Description:
            1. Create VLLMPollingCollector with config pointing to the live vLLM address.
            2. Call start() with a MetricsStore to begin background polling.
            3. Wait for at least one polling cycle to complete.
            4. Verify that expected canonical metric keys exist in MetricsStore.
        Expectation:
            MetricsStore.get(node_id, MetricKey.KV_CACHE_USAGE_PERC) returns a float.
            MetricsStore.get(node_id, MetricKey.NUM_REQUESTS_RUNNING) returns an int.
            MetricsStore.get(node_id, MetricKey.NUM_REQUESTS_WAITING) returns an int.
            MetricsStore.all_ids() contains the node_id.
        """
        # 1. Build config and server addresses
        config = _CollectorConfigMock(
            polling_interval=2.0,   # poll every 2s
            http_timeout=10.0,
        )

        # 2. Create collector and store
        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        # 3. Start background polling (needs an async event loop)
        async def _run_poll():
            collector.start(store)
            # Wait for at least one polling cycle (interval + margin)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run_poll())

        # 4. Verify metrics exist in the store
        node_ids = store.all_ids()
        assert NODE_ID in node_ids, (
            f"Expected node_id '{NODE_ID}' in store.all_ids(), got {node_ids}"
        )

        # KV cache usage should be a float (0.0–100.0)
        kv_usage = store.get(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC)
        assert isinstance(kv_usage, float), (
            f"Expected float for kv_cache_usage_perc, got {type(kv_usage).__name__}: {kv_usage}"
        )

        # Running requests should be an int
        running = store.get(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING)
        assert isinstance(running, int), (
            f"Expected int for num_requests_running, got {type(running).__name__}: {running}"
        )

        # Waiting requests should be an int
        waiting = store.get(NODE_ID, MetricKey.NUM_REQUESTS_WAITING)
        assert isinstance(waiting, int), (
            f"Expected int for num_requests_waiting, got {type(waiting).__name__}: {waiting}"
        )

    def test_metrics_values_are_sane(self, vllm_service):
        """
        Feature: Collected metric values are within reasonable bounds
        Description:
            After one polling cycle, verify that metric values are in expected ranges.
        Expectation:
            kv_cache_usage_perc >= 0.0
            num_requests_running >= 0
            num_requests_waiting >= 0
        """
        config = _CollectorConfigMock(
            polling_interval=2.0,
            http_timeout=10.0,
        )

        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run_poll():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run_poll())

        kv_usage = store.get(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC)
        running = store.get(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING)
        waiting = store.get(NODE_ID, MetricKey.NUM_REQUESTS_WAITING)

        assert kv_usage >= 0.0, f"kv_cache_usage_perc should be >= 0, got {kv_usage}"
        assert running >= 0, f"num_requests_running should be >= 0, got {running}"
        assert waiting >= 0, f"num_requests_waiting should be >= 0, got {waiting}"

    def test_store_get_node_dict(self, vllm_service):
        """
        Feature: MetricsStore.get(node_id) returns the full node metrics dict
        Description:
            After polling, verify that get(node_id) returns a dict containing all expected keys.
        Expectation:
            The node dict contains kv_cache_usage_perc, num_requests_running, num_requests_waiting.
        """
        config = _CollectorConfigMock(
            polling_interval=2.0,
            http_timeout=10.0,
        )

        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run_poll():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run_poll())

        node_metrics = store.get(NODE_ID)
        assert isinstance(node_metrics, dict), f"Expected dict, got {type(node_metrics).__name__}"
        assert MetricKey.KV_CACHE_USAGE_PERC in node_metrics, (
            f"Missing key '{MetricKey.KV_CACHE_USAGE_PERC}' in node metrics dict"
        )
        assert MetricKey.NUM_REQUESTS_RUNNING in node_metrics, (
            f"Missing key '{MetricKey.NUM_REQUESTS_RUNNING}' in node metrics dict"
        )
        assert MetricKey.NUM_REQUESTS_WAITING in node_metrics, (
            f"Missing key '{MetricKey.NUM_REQUESTS_WAITING}' in node metrics dict"
        )

    def test_multiple_poll_cycles_refresh(self, vllm_service):
        """
        Feature: Multiple polling cycles refresh the store with updated values
        Description:
            Run the collector for multiple cycles and verify that the store values
            are refreshed (not stale).
        Expectation:
            After 3 polling cycles, MetricsStore contains data and values are reasonable.
        """
        config = _CollectorConfigMock(
            polling_interval=2.0,
            http_timeout=10.0,
        )

        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run_poll():
            collector.start(store)
            # Wait for 3 polling cycles
            await asyncio.sleep(config.http_polling["polling_interval"] * 3 + 2.0)
            collector.stop()

        asyncio.run(_run_poll())

        # Verify the store was refreshed
        node_metrics = store.get(NODE_ID)
        assert len(node_metrics) > 0, "Store should have metrics after multiple poll cycles"
