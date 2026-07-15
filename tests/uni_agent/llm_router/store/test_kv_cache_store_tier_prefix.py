"""CPU unit tests for layer-aware prefix matching in KVCacheStore."""

from __future__ import annotations

import pytest

from uni_agent.llm_router.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.types import Layer
from uni_agent.llm_router.utils.hash import get_prefix_hashes

pytestmark = [pytest.mark.ut, pytest.mark.cpu]

BLOCK_SIZE = 4


def _store_with_block_size() -> KVCacheStore:
    store = KVCacheStore()
    store.block_size = BLOCK_SIZE
    return store


def _hashes(prompt_ids: list[int]) -> list[str]:
    return [str(h) for h in get_prefix_hashes(prompt_ids, BLOCK_SIZE)]


def test_store_keeps_gpu_and_cpu_indexes_separate() -> None:
    store = _store_with_block_size()
    store.add_blocks("replica-1", ["hash-1"], layer=Layer.GPU)
    store.add_blocks("replica-1", ["hash-1"], layer=Layer.CPU)

    assert store.replicas_by_block == {"hash-1": {"replica-1"}}
    assert store._replicas_by_layer[Layer.CPU] == {"hash-1": {"replica-1"}}

    store.remove_blocks("replica-1", ["hash-1"], layer=Layer.CPU)
    assert store.replicas_by_block == {"hash-1": {"replica-1"}}
    assert store._replicas_by_layer[Layer.CPU] == {}

    store.clear_replica("replica-1")
    assert store.replicas_by_block == {}


def test_store_computes_cpu_contiguous_prefix_hit_rate() -> None:
    prompt_ids = list(range(1, 9))
    h1, h2 = _hashes(prompt_ids)
    store = _store_with_block_size()
    store.add_blocks("replica-1", [h1], layer=Layer.CPU)
    store.add_blocks("replica-2", [h1, h2], layer=Layer.CPU)

    assert store.get_layer_prefix_hit_rate("replica-1", prompt_ids, Layer.CPU) == 0.5
    assert store.get_layer_prefix_hit_rate("replica-2", prompt_ids, Layer.CPU) == 1.0
    assert store.get_layer_prefix_hit_rate("unknown-replica", prompt_ids, Layer.CPU) == 0.0
    assert store.get_layer_prefix_hit_rate("replica-1", prompt_ids, Layer.SSD) == 0.0


def test_cpu_later_block_without_first_block_is_not_credited() -> None:
    prompt_ids = list(range(1, 13))
    _, h2, h3 = _hashes(prompt_ids)
    store = _store_with_block_size()
    store.add_blocks("replica-1", [h2, h3], layer=Layer.CPU)

    assert store.get_layer_prefix_hit_rate("replica-1", prompt_ids, Layer.CPU) == 0.0


def test_cpu_index_does_not_affect_gpu_prefix_hit_rate() -> None:
    prompt_ids = list(range(1, 9))
    h1, _ = _hashes(prompt_ids)
    store = _store_with_block_size()
    store.add_blocks("replica-1", [h1], layer=Layer.CPU)

    assert store.get_layer_prefix_hit_rate("replica-1", prompt_ids, Layer.GPU) == 0.0


def test_duplicate_events_do_not_drift_layer_counts() -> None:
    store = _store_with_block_size()
    store.add_blocks("replica-1", ["hash-1"], layer=Layer.GPU)
    store.add_blocks("replica-1", ["hash-1"], layer=Layer.GPU)
    assert store.per_replica_block_counts() == {"replica-1": 1}

    store.remove_blocks("replica-1", ["hash-1"], layer=Layer.GPU)
    store.remove_blocks("replica-1", ["hash-1"], layer=Layer.GPU)
    assert store.per_replica_block_counts() == {}
