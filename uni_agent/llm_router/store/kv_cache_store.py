"""KVCacheStore — backend-agnostic data carrier for KV cache mapping tables."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from uni_agent.llm_router.utils.hash import get_prefix_hashes


class KVCacheStore:
    """Mutable data carrier for KV cache mapping tables.

    Attributes:
        block_size: Learned block size (None until first BlockStored event).
        replicas_by_block: local prefix hash → set of replica_ids that
            cache it.  Aligns with aibrix prefixMap (hash → pods).
    """

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self.replicas_by_block: dict[str, set[str]] = {}
        # Incremental per-replica retained-block count, maintained alongside
        # ``replicas_by_block`` so ``per_replica_block_counts()`` is O(replicas)
        # (not O(blocks)) and holds the lock briefly.
        self._replica_block_count: dict[str, int] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear all blocks for a replica from the reverse index.

        Iterates ``replicas_by_block`` to remove the replica from every
        block entry, then deletes empty entries.  O(n) in the number of
        unique blocks, but replica count is typically small (< 100).
        """
        with self._lock:
            stale_hashes: list[str] = []
            for bh, replicas in self.replicas_by_block.items():
                if replica_id not in replicas:
                    continue
                replicas.discard(replica_id)
                if not replicas:
                    stale_hashes.append(bh)
            for bh in stale_hashes:
                del self.replicas_by_block[bh]
            self._replica_block_count.pop(replica_id, None)

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str]) -> None:
        """Add blocks to a replica, updating the reverse index."""
        with self._lock:
            for bh in block_hashes:
                if bh not in self.replicas_by_block:
                    self.replicas_by_block[bh] = set()
                replicas = self.replicas_by_block[bh]
                if replica_id not in replicas:
                    replicas.add(replica_id)
                    self._replica_block_count[replica_id] = self._replica_block_count.get(replica_id, 0) + 1

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str]) -> None:
        """Remove blocks from a replica, updating the reverse index."""
        with self._lock:
            for bh in block_hashes:
                replicas = self.replicas_by_block.get(bh)
                if replicas is not None and replica_id in replicas:
                    replicas.discard(replica_id)
                    self._replica_block_count[replica_id] = self._replica_block_count.get(replica_id, 0) - 1
                    if not replicas:
                        del self.replicas_by_block[bh]

    # ── Retained-cache size ─────────────────────────────────────────────

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct prefix blocks it retains}``.

        This is the **retained-cache size per replica** — vllm exposes no
        direct gauge for this (``kv_cache_usage_perc`` is running-only;
        retained cache blocks live in the free pool, hash-marked for reuse).
        Maintained incrementally alongside ``replicas_by_block`` in
        add_blocks/remove_blocks/clear_replica, so this is O(replicas) and
        holds the lock only briefly (does not scan all blocks).
        Divide by the per-replica block pool (~11929 for our config) to get a
        retained-cache occupancy that, unlike kv_cache_usage_perc, rises as
        distinct prefixes accumulate and signals impending LRU eviction.
        """
        with self._lock:
            return dict(self._replica_block_count)

    # ── Prefix hit rate queries ─────────────────────────────────────────

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Match prefix hashes against cached blocks, return per-replica hit percent.

        Algorithm (matching aibrix MatchPrefix):
            1. Compute prefix hashes via get_prefix_hashes
            2. For each prefix hash, check replicas_by_block to find replicas
            3. Stop at first hash where no replica matches (chain break)
            4. Compute percent = (matched_count * 100) // total_hashes

        The entire computation runs inside the lock so the snapshot is
        consistent — no partial reads interleaved with concurrent writes.

        Args:
            prompt_ids: Current request's prompt token IDs.

        Returns:
            Dict of replica_id → prefix_match_percent (0–100).
            Empty dict if block_size is unknown or no full blocks.
        """
        with self._lock:
            if self.block_size is None:
                return {}

            prefix_hashes = get_prefix_hashes(prompt_ids, self.block_size)
            if not prefix_hashes:
                return {}

            hash_strs = [str(h) for h in prefix_hashes]

            # Sequential prefix matching (aibrix MatchPrefix pattern)
            prefix_match_replicas: dict[str, int] = {}

            for i, hs in enumerate(hash_strs):
                cached_replicas = self.replicas_by_block.get(hs)
                if cached_replicas is None or len(cached_replicas) == 0:
                    break  # chain break — no replica caches this hash

                prefix_match_percent = (i + 1) * 100 // len(hash_strs)

                for replica_id in cached_replicas:
                    prefix_match_replicas[replica_id] = prefix_match_percent

            return prefix_match_replicas

    def get_tier_prefix_hit_rate(
        self,
        node_id: str,
        prompt_ids: list[int],
        tier: str,
    ) -> float:
        """Query tier-level prefix cache hit rate (slow-path data).

        v1: placeholder — returns 0.0 when tier metrics are not yet available.
        v2: will call Mooncake /batch_query_keys API for real-time query.

        Args:
            node_id: Target node.
            prompt_ids: Current request's prompt token IDs.
            tier: "cpu" or "ssd".

        Returns:
            Hit rate 0.0–1.0; returns 0.0 if no data in snapshot.
        """
        # v1 placeholder — read from snapshot when tier metrics are available
        return 0.0
