"""
KVCacheStore — backend-agnostic data carrier for KV cache mapping tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class KVCacheStore:
    """Mutable data carrier for KV cache mapping tables.

    This is the single source of truth for all cross-replica KV cache
    state. 

    Attributes:
        block_size: Learned block size (``None`` until first BlockStored event).
        replicas_by_block: local prefix hash → set of replica_ids that
            cache it.  Aligns with aibrix's ``prefixMap`` (hash → pods).
    """

    block_size: int | None = None
    replicas_by_block: dict[str, set[str]] = field(default_factory=dict)

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear all blocks for a replica from the reverse index.

        Iterates ``replicas_by_block`` to remove the replica from every
        block entry, then deletes empty entries.  O(n) in the number of
        unique blocks, but replica count is typically small (< 100).
        """
        stale_hashes: list[str] = []
        for bh, replicas in self.replicas_by_block.items():
            if replica_id in replicas:
                replicas.discard(replica_id)
                if not replicas:
                    stale_hashes.append(bh)
        for bh in stale_hashes:
            del self.replicas_by_block[bh]

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str]) -> None:
        """Add blocks to a replica, updating the reverse index."""
        for bh in block_hashes:
            if bh not in self.replicas_by_block:
                self.replicas_by_block[bh] = set()
            self.replicas_by_block[bh].add(replica_id)

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str]) -> None:
        """Remove blocks from a replica, updating the reverse index."""
        for bh in block_hashes:
            if bh in self.replicas_by_block:
                self.replicas_by_block[bh].discard(replica_id)
                if not self.replicas_by_block[bh]:
                    del self.replicas_by_block[bh]
