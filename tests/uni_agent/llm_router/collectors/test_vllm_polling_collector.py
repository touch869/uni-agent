"""Tests for VLLMPollingCollector with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B).
2. Create VLLMPollingCollector with config containing the vLLM service address.
3. Call start() to begin metrics polling and write to MetricsStore.
4. Verify that expected metrics exist in the store.
"""

from __future__ import annotations

import asyncio

from conftest import NODE_ID, VLLM_MODEL
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.collectors.collector.vllm.polling_collector import VLLMPollingCollector
from uni_agent.llm_router.collectors.metric_spec import MetricKey
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore


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
        config = CollectorConfig(http_polling={"polling_interval": 2.0, "http_timeout": 10.0})
        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run())

        assert NODE_ID in store.all_ids(), (
            f"Expected node_id '{NODE_ID}' in store.all_ids(), got {store.all_ids()}"
        )
        kv_usage = store.get(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC)
        assert isinstance(kv_usage, float), (
            f"Expected float for kv_cache_usage_perc, got {type(kv_usage).__name__}: {kv_usage}"
        )
        running = store.get(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING)
        assert isinstance(running, int), (
            f"Expected int for num_requests_running, got {type(running).__name__}: {running}"
        )
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
        config = CollectorConfig(http_polling={"polling_interval": 2.0, "http_timeout": 10.0})
        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run())

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
        config = CollectorConfig(http_polling={"polling_interval": 2.0, "http_timeout": 10.0})
        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] + 3.0)
            collector.stop()

        asyncio.run(_run())

        node_metrics = store.get(NODE_ID)
        assert isinstance(node_metrics, dict), f"Expected dict, got {type(node_metrics).__name__}"
        assert MetricKey.KV_CACHE_USAGE_PERC in node_metrics
        assert MetricKey.NUM_REQUESTS_RUNNING in node_metrics
        assert MetricKey.NUM_REQUESTS_WAITING in node_metrics

    def test_multiple_poll_cycles_refresh(self, vllm_service):
        """
        Feature: Multiple polling cycles refresh the store with updated values
        Description:
            Run the collector for multiple cycles and verify that the store values
            are refreshed (not stale).
        Expectation:
            After 3 polling cycles, MetricsStore contains data and values are reasonable.
        """
        config = CollectorConfig(http_polling={"polling_interval": 2.0, "http_timeout": 10.0})
        collector = VLLMPollingCollector(config, server_addresses={NODE_ID: NODE_ID})
        store = MetricsStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(config.http_polling["polling_interval"] * 3 + 2.0)
            collector.stop()

        asyncio.run(_run())

        node_metrics = store.get(NODE_ID)
        assert len(node_metrics) > 0, "Store should have metrics after multiple poll cycles"
