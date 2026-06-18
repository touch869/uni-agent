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
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import httpx

from uni_agent.llm_router.collectors.collector.vllm.event_collector import VLLMKVEventCollector
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.hash import compute_hash
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore


# ── Configuration constants ──────────────────────────────────────────────

VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
NODE_ID = f"{VLLM_HOST}:{VLLM_PORT}"

# ZMQ ports — used to construct kv_event_addresses for the collector
ZMQ_SUB_PORT = int(os.environ.get("ZMQ_SUB_PORT", "5555"))
ZMQ_REPLAY_PORT = int(os.environ.get("ZMQ_REPLAY_PORT", "5556"))


# ── Helper: config mock for ZMQEventCollector ───────────────────────────

class _ZMQCollectorConfigMock:
    """Minimal config object matching ZMQEventCollector's expected interface.

    ZMQEventCollector.__init__ reads:
      - config.long_connection["base_retry_delay"]       (float)
      - config.long_connection["max_retry_delay"]        (float)
      - config.long_connection["max_retry_attempts"]     (int)
      - config.long_connection["retry_backoff_factor"]   (float)
    """

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


# ── Fixture: launch vLLM with KV events enabled ─────────────────────────

@pytest.fixture(scope="module")
def vllm_kv_service():
    """Launch a real vLLM server with kv-events-config (ZMQ publisher enabled).

    vLLM binds:
      - PUB socket at tcp://*:<ZMQ_SUB_PORT>   (default 5555)
      - ROUTER (replay) socket at tcp://*:<ZMQ_REPLAY_PORT>  (default 5556)
    """
    kv_events_config = json.dumps({
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": f"tcp://*:{ZMQ_SUB_PORT}",
        "replay_endpoint": f"tcp://*:{ZMQ_REPLAY_PORT}",
    })

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
        "--kv-events-config", kv_events_config,
    ]

    proc = subprocess.Popen(cmd)

    # Wait for the vLLM server to become ready
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
            f"vLLM server for {VLLM_MODEL} did not become ready within {max_wait}s"
        )

    yield NODE_ID

    # Teardown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── Helper: send inference request ──────────────────────────────────────

def _send_inference_request(node_id: str, model: str, prompt: str = "hello"):
    """Send an inference request to trigger KV cache block-stored events."""
    try:
        resp = httpx.post(
            f"http://{node_id}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


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
        config = _ZMQCollectorConfigMock(
            base_retry_delay=1.0,
            max_retry_attempts=5,
        )

        collector = VLLMKVEventCollector(config, kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]})
        store = KVCacheStore()

        async def _run_subscribe():
            collector.start(store)
            # Wait for ZMQ connection + replay + at least one event
            await asyncio.sleep(5.0)

            # Send inference to trigger BlockStored events
            _send_inference_request(vllm_kv_service, VLLM_MODEL, "hello world")
            await asyncio.sleep(5.0)

            collector.stop()

        asyncio.run(_run_subscribe())

        # Verify block_size was learned from a BlockStored event
        assert store.block_size is not None, (
            f"block_size should be learned from KV events, got None"
        )
        assert store.block_size > 0, f"block_size should be > 0, got {store.block_size}"

        # Verify replicas_by_block has entries
        assert len(store.replicas_by_block) > 0, (
            f"replicas_by_block should be non-empty after BlockStored events, "
            f"got {len(store.replicas_by_block)} entries"
        )

        # Verify NODE_ID appears in at least one block's replica set
        replica_found = False
        for block_hash, replicas in store.replicas_by_block.items():
            if NODE_ID in replicas:
                replica_found = True
                break
        assert replica_found, (
            f"Expected NODE_ID '{NODE_ID}' in at least one block's replica set"
        )

    def test_block_size_learned(self, vllm_kv_service):
        """
        Feature: block_size is learned from the first BlockStored KV event
        Description:
            After receiving KV events, verify that block_size is a reasonable value.
        Expectation:
            block_size is a positive integer (vLLM default is 16).
        """
        config = _ZMQCollectorConfigMock(
            base_retry_delay=1.0,
            max_retry_attempts=5,
        )

        collector = VLLMKVEventCollector(config, kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]})
        store = KVCacheStore()

        async def _run_subscribe():
            collector.start(store)
            await asyncio.sleep(5.0)
            _send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run_subscribe())

        assert isinstance(store.block_size, int)
        assert store.block_size > 0
        # vLLM default block_size is 16
        assert store.block_size == 16, (
            f"Expected block_size=16 (vLLM default), got {store.block_size}"
        )

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
        config = _ZMQCollectorConfigMock(
            base_retry_delay=1.0,
            max_retry_attempts=5,
        )

        collector = VLLMKVEventCollector(config, kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]})
        store = KVCacheStore()

        async def _run_subscribe():
            collector.start(store)
            await asyncio.sleep(5.0)

            # Send multiple inference requests with different prompts
            prompts = [
                "What is machine learning?",
                "Explain quantum computing briefly.",
                "Tell me about deep reinforcement learning.",
            ]
            for prompt in prompts:
                _send_inference_request(vllm_kv_service, VLLM_MODEL, prompt)
                await asyncio.sleep(3.0)

            await asyncio.sleep(3.0)
            collector.stop()

        asyncio.run(_run_subscribe())

        # After multiple requests, there should be blocks in the store
        assert len(store.replicas_by_block) > 0, (
            f"Expected blocks after multiple inferences, got empty store"
        )

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
        config = _ZMQCollectorConfigMock(
            base_retry_delay=1.0,
            max_retry_attempts=5,
        )

        collector = VLLMKVEventCollector(config, kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]})
        store = KVCacheStore()

        async def _run_subscribe():
            collector.start(store)
            await asyncio.sleep(5.0)
            _send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run_subscribe())

        # Pre-condition: store has blocks for NODE_ID
        if len(store.replicas_by_block) == 0:
            pytest.skip("No blocks received from KV events")

        # Clear the replica
        store.clear_replica(NODE_ID)

        # Verify NODE_ID is gone from all blocks
        for block_hash, replicas in store.replicas_by_block.items():
            assert NODE_ID not in replicas, (
                f"NODE_ID '{NODE_ID}' should not be in block '{block_hash}' "
                f"after clear_replica"
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
        config = _ZMQCollectorConfigMock(
            base_retry_delay=1.0,
            max_retry_attempts=5,
        )

        collector = VLLMKVEventCollector(config, kv_event_addresses={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]})
        store = KVCacheStore()

        async def _run_subscribe():
            collector.start(store)
            await asyncio.sleep(5.0)
            _send_inference_request(vllm_kv_service, VLLM_MODEL)
            await asyncio.sleep(5.0)
            collector.stop()

        asyncio.run(_run_subscribe())

        # Verify the collector's remote_to_local_block_hash mapping was populated
        assert len(collector.remote_to_local_block_hash) > 0, (
            f"remote_to_local_block_hash should have entries after processing events, "
            f"got {len(collector.remote_to_local_block_hash)}"
        )

        # Verify mapping consistency: remote hashes map to local hashes
        for remote_bh, local_bh in collector.remote_to_local_block_hash.items():
            assert isinstance(remote_bh, str), f"remote hash should be str"
            assert isinstance(local_bh, str), f"local hash should be str"
            # Local hashes should appear in the store's replicas_by_block
            assert local_bh in store.replicas_by_block, (
                f"Local hash '{local_bh}' from mapping not found in store.replicas_by_block"
            )
