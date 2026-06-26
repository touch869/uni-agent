"""Unit tests for normalized vLLM KV-cache event parsing."""

from __future__ import annotations

import struct

from uni_agent.llm_router.collectors.collector.vllm.event_collector import (
    VLLMKVEventCollector,
)
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.hash import compute_hash, get_prefix_hashes
from uni_agent.llm_router.collectors.provider import RouteDataProvider
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore


def test_gpu_block_stored_parses_medium_and_tokens() -> None:
    raw = [
        0.0,
        [[0, ["hash-1"], None, [1, 2, 3, 4], 4, None, "GPU", None]],
    ]

    events = KVCacheEvent.from_raw(raw, default_replica_id="replica-1")

    assert len(events) == 1
    event = events[0]
    assert event.medium == "gpu"
    assert event.block_size == 4
    assert event.token_ids is not None
    assert len(event.token_ids) == 1


def test_cpu_block_stored_allows_zero_block_size() -> None:
    raw = [
        0.0,
        [[0, ["hash-1"], None, [], 0, None, "CPU", None]],
    ]

    events = KVCacheEvent.from_raw(raw, default_replica_id="replica-1")

    assert len(events) == 1
    event = events[0]
    assert event.medium == "cpu"
    assert event.block_size == 0
    assert event.token_ids is None


def test_bytes_block_hash_normalizes_to_vllm_int_hash() -> None:
    raw_hash = bytes(range(32))
    vllm_int_hash = int.from_bytes(raw_hash, byteorder="big") & ((1 << 64) - 1)
    raw = [
        0.0,
        [
            [0, [vllm_int_hash], None, [1, 2, 3, 4], 4, None, "GPU", None],
            [0, [raw_hash], None, [], 4, None, "CPU", None],
        ],
    ]

    events = KVCacheEvent.from_raw(raw, default_replica_id="replica-1")

    assert len(events) == 2
    assert events[0].block_hashes == [str(vllm_int_hash)]
    assert events[1].block_hashes == [str(vllm_int_hash)]


def test_cpu_block_removed_parses_medium() -> None:
    raw = [0.0, [[1, ["hash-1"], "CPU", None]]]

    events = KVCacheEvent.from_raw(raw, default_replica_id="replica-1")

    assert len(events) == 1
    event = events[0]
    assert event.is_remove
    assert event.medium == "cpu"


def test_store_keeps_gpu_and_cpu_tables_separate() -> None:
    store = KVCacheStore()
    store.add_blocks("replica-1", ["hash-1"], tier="gpu")
    store.add_blocks("replica-1", ["hash-1"], tier="cpu")
    store.remove_blocks("replica-1", ["hash-1"], tier="cpu")

    assert store.replicas_by_block == {"hash-1": {"replica-1"}}
    assert store.replicas_by_tier_and_block == {
        ("gpu", "hash-1"): {"replica-1"}
    }
    assert store.cpu_tracking_replicas == {"replica-1"}

    store.clear_replica("replica-1")
    assert store.replicas_by_block == {}
    assert store.replicas_by_tier_and_block == {}


def test_collector_routes_cpu_events_without_mutating_gpu_table() -> None:
    collector = VLLMKVEventCollector.__new__(VLLMKVEventCollector)
    collector._store = KVCacheStore()
    collector.remote_to_local_block_hash = {}
    collector.cpu_remote_to_local_block_hash = {}

    collector._apply_event(KVCacheEvent(
        event_type="stored",
        replica_id="replica-1",
        block_hashes=["remote-1"],
        parent_block_hash=None,
        token_ids=[struct.pack(">4I", 1, 2, 3, 4)],
        block_size=4,
        medium="gpu",
    ))
    local_hash = next(iter(collector._store.replicas_by_block))

    collector._apply_event(KVCacheEvent(
        event_type="stored",
        replica_id="replica-1",
        block_hashes=["remote-1"],
        parent_block_hash=None,
        token_ids=None,
        block_size=0,
        medium="cpu",
    ))
    assert collector._store.replicas_by_tier_and_block[("cpu", local_hash)] == {
        "replica-1"
    }

    collector._apply_event(KVCacheEvent(
        event_type="removed",
        replica_id="replica-1",
        block_hashes=["remote-1"],
        parent_block_hash=None,
        token_ids=None,
        block_size=None,
        medium="gpu",
    ))
    assert collector._store.replicas_by_block == {}
    assert collector._store.replicas_by_tier_and_block[("cpu", local_hash)] == {
        "replica-1"
    }

    collector._apply_event(KVCacheEvent(
        event_type="removed",
        replica_id="replica-1",
        block_hashes=["remote-1"],
        parent_block_hash=None,
        token_ids=None,
        block_size=None,
        medium="cpu",
    ))
    assert collector._store.replicas_by_tier_and_block == {}


def test_collector_routes_cpu_bytes_hash_through_gpu_int_mapping() -> None:
    collector = VLLMKVEventCollector.__new__(VLLMKVEventCollector)
    collector._store = KVCacheStore()
    collector.remote_to_local_block_hash = {}
    collector.cpu_remote_to_local_block_hash = {}
    raw_hash = bytes(range(32))
    vllm_int_hash = str(int.from_bytes(raw_hash, byteorder="big") & ((1 << 64) - 1))

    gpu_event, cpu_event = KVCacheEvent.from_raw(
        [
            0.0,
            [
                [0, [int(vllm_int_hash)], None, [1, 2, 3, 4], 4, None, "GPU", None],
                [0, [raw_hash], None, [], 4, None, "CPU", None],
            ],
        ],
        default_replica_id="replica-1",
    )

    collector._apply_event(gpu_event)
    local_hash = next(iter(collector._store.replicas_by_block))
    collector._apply_event(cpu_event)

    assert collector._store.replicas_by_tier_and_block[("cpu", local_hash)] == {
        "replica-1"
    }
    assert collector.cpu_remote_to_local_block_hash[("replica-1", vllm_int_hash)] == local_hash


def test_collector_computes_cpu_hashes_from_token_ids() -> None:
    collector = VLLMKVEventCollector.__new__(VLLMKVEventCollector)
    collector._store = KVCacheStore()
    collector.remote_to_local_block_hash = {}
    collector.cpu_remote_to_local_block_hash = {}
    block_bytes = struct.pack(">4I", 1, 2, 3, 4)

    collector._apply_event(KVCacheEvent(
        event_type="stored",
        replica_id="replica-1",
        block_hashes=["cpu-remote-1"],
        parent_block_hash=None,
        token_ids=[block_bytes],
        block_size=4,
        medium="cpu",
    ))

    local_hash = str(compute_hash(0, block_bytes, seed=0))
    assert collector._store.replicas_by_tier_and_block[("cpu", local_hash)] == {
        "replica-1"
    }
    assert collector.cpu_remote_to_local_block_hash[("replica-1", "cpu-remote-1")] == local_hash
    assert collector.remote_to_local_block_hash == {}


def test_provider_computes_gpu_prefix_hit_rate_from_unified_table() -> None:
    store = KVCacheStore(block_size=4)
    prompt_ids = list(range(1, 9))
    prefix_hashes = get_prefix_hashes(prompt_ids, store.block_size)
    store.add_blocks("replica-1", [str(prefix_hashes[0])], tier="gpu")

    provider = RouteDataProvider.__new__(RouteDataProvider)
    provider._stores = {KVCacheStore: store}

    assert provider.get_gpu_prefix_hit_rate(prompt_ids) == {"replica-1": 50}


def test_provider_computes_cpu_contiguous_prefix_hit_rate() -> None:
    store = KVCacheStore(block_size=4)
    prompt_ids = list(range(1, 9))
    prefix_hashes = get_prefix_hashes(prompt_ids, store.block_size)
    store.add_blocks("replica-1", [str(prefix_hashes[0])], tier="cpu")

    provider = RouteDataProvider.__new__(RouteDataProvider)
    provider._stores = {KVCacheStore: store}

    assert provider.get_tier_prefix_hit_rate(
        "replica-1", prompt_ids, "cpu"
    ) == 0.5
    assert provider.get_tier_prefix_hit_rate(
        "unknown-replica", prompt_ids, "cpu"
    ) is None
    assert provider.get_tier_prefix_hit_rate(
        "replica-1", prompt_ids, "ssd"
    ) is None


def test_cpu_remove_does_not_mark_replica_as_tracked() -> None:
    """A CPU BlockRemoved on a replica that never stored CPU data must not
    mark that replica as having CPU-tier data (which would make the provider
    report a 0.0 hit rate instead of ``None`` for "data unavailable")."""
    store = KVCacheStore()
    store.remove_blocks("gpu-only-replica", ["hash-1"], tier="cpu")

    assert "gpu-only-replica" not in store.cpu_tracking_replicas
    assert store.replicas_by_tier_and_block == {}
