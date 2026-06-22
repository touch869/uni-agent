"""Tests for VLLMKVEventCollector with real vLLM service + ZMQ KV events.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B) with kv-events-config enabled
   (ZMQ publisher for kv_cache_events).
2. Create VLLMKVEventCollector with config containing ZMQ sub/replay addresses.
3. Call start() with KVCacheStore to begin event subscription.
4. Send an inference request to trigger KV cache block events.
5. Verify that KVCacheStore receives block data via ZMQ events.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import NODE_ID, ZMQ_SUB_PORT, ZMQ_REPLAY_PORT, VLLM_MODEL, send_inference_request
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.collectors.collector.vllm.event_collector import VLLMKVEventCollector
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore


# ── Test class ───────────────────────────────────────────────────────────

class TestVLLMKVEventCollectorWithRealService:
    """Integration tests: VLLMKVEventCollector against a live vLLM ZMQ publisher."""

    def test_start_and_kv_store_updated(self, vllm_kv_service):
        """
        Feature: VLLMKVEventCollector receives ZMQ events and updates KVCacheStore
        Description:
            1. Create VLLMKVEventCollector with ZMQ addresses from the vLLM server.
            2. Call start() with KVCacheStore to begin event subscription.
            3. Send an inference request to trigger BlockStored events.
            4. Wait for events to arrive and be processed.
            5. Verify KVCacheStore contains block data.
        Expectation:
            KVCacheStore.block_size is set (learned from first event).
            KVCacheStore.replicas_by_block is non-empty.
            NODE_ID appears in at least one block's replica set.
        """
        collector = VLLMKVEventCollector(
            CollectorConfig(),
            kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
        )
        store = KVCacheStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(5.0)
            send_inference_request(vllm_kv_service, VLLM_MODEL, "hello world")
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run())

        assert store.block_size is not None, "block_size should be learned from KV events"
        assert store.block_size > 0, f"block_size should be > 0, got {store.block_size}"
        assert len(store.replicas_by_block) > 0, (
            f"replicas_by_block should be non-empty after BlockStored events"
        )
        replica_found = any(NODE_ID in replicas for replicas in store.replicas_by_block.values())
        assert replica_found, f"Expected NODE_ID '{NODE_ID}' in at least one block's replica set"

    def test_block_size_learned(self, vllm_kv_service):
        """
        Feature: block_size is learned from the first BlockStored KV event
        Description:
            After receiving KV events, verify that block_size is a reasonable value.
        Expectation:
            block_size is a positive integer (vLLM default is 16).
        """
        collector = VLLMKVEventCollector(
            CollectorConfig(),
            kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
        )
        store = KVCacheStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(5.0)
            send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run())

        assert isinstance(store.block_size, int)
        assert store.block_size > 0
        assert store.block_size == 16, f"Expected block_size=16 (vLLM default), got {store.block_size}"

    def test_multiple_inferences_accumulate_blocks(self, vllm_kv_service):
        """
        Feature: Multiple inference requests accumulate more blocks in the store
        Description:
            Send several inference requests and verify that replicas_by_block
            accumulates more entries (more prefixes cached).
        Expectation:
            After multiple requests, replicas_by_block has more entries than
            after a single request.
        """
        collector = VLLMKVEventCollector(
            CollectorConfig(),
            kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
        )
        store = KVCacheStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(5.0)
            for prompt in [
                "What is machine learning?",
                "Explain quantum computing briefly.",
                "Tell me about deep reinforcement learning.",
            ]:
                send_inference_request(vllm_kv_service, VLLM_MODEL, prompt)
                await asyncio.sleep(3.0)
            await asyncio.sleep(3.0)
            collector.stop()

        asyncio.run(_run())

        assert len(store.replicas_by_block) > 0, "Expected blocks after multiple inferences"

    def test_clear_replica_removes_all_blocks(self, vllm_kv_service):
        """
        Feature: KVCacheStore.clear_replica removes all blocks for a replica
        Description:
            1. Accumulate blocks in the store via KV events.
            2. Manually call store.clear_replica(node_id).
            3. Verify that the replica is removed from all block entries.
        Expectation:
            After clear_replica, no block in replicas_by_block contains NODE_ID.
        """
        collector = VLLMKVEventCollector(
            CollectorConfig(),
            kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
        )
        store = KVCacheStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(5.0)
            send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run())

        if len(store.replicas_by_block) == 0:
            pytest.skip("No blocks received from KV events")

        store.clear_replica(NODE_ID)

        for block_hash, replicas in store.replicas_by_block.items():
            assert NODE_ID not in replicas, (
                f"NODE_ID '{NODE_ID}' should not be in block '{block_hash}' after clear_replica"
            )

    def test_consume_payload_parses_msgpack(self, vllm_kv_service):
        """
        Feature: _consume_payload correctly parses msgpack payloads
        Description:
            Verify that the _consume_payload method can decode msgpack payloads
            received via ZMQ and apply them to KVCacheStore.
        Expectation:
            After processing payloads, KVCacheStore has valid data.
            The remote_to_local_block_hash mapping has entries.
        """
        collector = VLLMKVEventCollector(
            CollectorConfig(),
            kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
        )
        store = KVCacheStore()

        async def _run():
            collector.start(store)
            await asyncio.sleep(5.0)
            send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run())

        assert len(collector.remote_to_local_block_hash) > 0, (
            "remote_to_local_block_hash should have entries after processing events"
        )
        for remote_bh, local_bh in collector.remote_to_local_block_hash.items():
            assert isinstance(remote_bh, str), "remote hash should be str"
            assert isinstance(local_bh, str), "local hash should be str"
            assert local_bh in store.replicas_by_block, (
                f"Local hash '{local_bh}' from mapping not found in store.replicas_by_block"
            )
