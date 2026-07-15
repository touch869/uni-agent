"""KVCacheStore — backend-agnostic data carrier for KV cache mapping tables."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from uni_agent.llm_router.types import Layer
from uni_agent.llm_router.utils.hash import get_prefix_hashes


class KVCacheStore:
    """Mutable data carrier for per-layer KV cache mapping tables.

    Attributes:
        block_size: Learned block size (None until first BlockStored event).
        replicas_by_block: Public GPU reverse index mapping local prefix hashes
            to the replica IDs that cache them. CPU and SSD use separate
            internal indexes addressed through the canonical ``Layer`` API.
    """

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self.replicas_by_block: dict[str, set[str]] = {}
        self._replicas_by_layer: dict[Layer, dict[str, set[str]]] = {
            Layer.GPU: self.replicas_by_block,
            Layer.CPU: {},
            Layer.SSD: {},
        }
        self._replica_layer_counts: dict[Layer, dict[str, int]] = {
            Layer.GPU: {},
            Layer.CPU: {},
            Layer.SSD: {},
        }
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear every cached layer entry for a replica.

        O(n) in the total number of unique blocks across all cache layers.
        """
        with self._lock:
            for layer_index in self._replicas_by_layer.values():
                stale_hashes: list[str] = []
                for block_hash, replicas in layer_index.items():
                    if replica_id not in replicas:
                        continue
                    replicas.discard(replica_id)
                    if not replicas:
                        stale_hashes.append(block_hash)
                for block_hash in stale_hashes:
                    del layer_index[block_hash]

            for layer_counts in self._replica_layer_counts.values():
                layer_counts.pop(replica_id, None)

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str],
                    layer: Layer = Layer.GPU) -> None:
        """Add distinct blocks to a replica in the selected cache layer."""
        with self._lock:
            layer_index = self._replicas_by_layer[layer]
            layer_counts = self._replica_layer_counts[layer]
            for block_hash in block_hashes:
                replicas = layer_index.setdefault(block_hash, set())
                if replica_id in replicas:
                    continue
                replicas.add(replica_id)
                layer_counts[replica_id] = layer_counts.get(replica_id, 0) + 1

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str],
                      layer: Layer = Layer.GPU) -> None:
        """Remove blocks from a replica at a layer, updating its reverse index."""
        with self._lock:
            layer_index = self._replicas_by_layer[layer]
            layer_counts = self._replica_layer_counts[layer]
            for block_hash in block_hashes:
                replicas = layer_index.get(block_hash)
                if replicas is None or replica_id not in replicas:
                    continue

                replicas.discard(replica_id)
                remaining = layer_counts.get(replica_id, 0) - 1
                if remaining > 0:
                    layer_counts[replica_id] = remaining
                else:
                    layer_counts.pop(replica_id, None)
                if not replicas:
                    del layer_index[block_hash]

    # ── Retained-cache size ─────────────────────────────────────────────

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct GPU prefix blocks it retains}``.

        GPU-only count — feeds the retained-load formula. Maintained incrementally,
        O(replicas). Divide by the per-replica block pool size for occupancy.
        """
        with self._lock:
            return dict(self._replica_layer_counts[Layer.GPU])

    # ── Prefix hit rate queries ─────────────────────────────────────────

    def get_layer_prefix_hit_rate(self, node_id: str, prompt_ids: list[int],
                                   layer: Layer = Layer.GPU) -> float:
        """Prefix-cache hit rate for a node at a layer, in ``[0.0, 1.0]``.

        Walk the selected layer's reverse index along the prompt's prefix-hash
        chain until a hash is not cached on this node.
        """
        with self._lock:
            if self.block_size is None:
                return 0.0
            prefix_hashes = get_prefix_hashes(prompt_ids, self.block_size)
            if not prefix_hashes:
                return 0.0
            layer_index = self._replicas_by_layer[layer]
            matched = 0
            for index, prefix_hash in enumerate(prefix_hashes):
                cached_replicas = layer_index.get(str(prefix_hash))
                if cached_replicas is None or node_id not in cached_replicas:
                    break
                matched = index + 1
            return matched / len(prefix_hashes)
