"""Tier-aware KV-cache mapping store."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class KVCacheStore:
    """Mutable data carrier for tiered KV-cache mapping tables.

    replicas_by_block remains a GPU-only compatibility view.
    """

    block_size: int | None = None
    replicas_by_tier_and_block: dict[tuple[str, str], set[str]] = field(
        default_factory=dict
    )

    @property
    def replicas_by_block(self) -> dict[str, set[str]]:
        """GPU-only compatibility view keyed by local prefix hash."""
        return {
            block_hash: replicas
            for (tier, block_hash), replicas in self.replicas_by_tier_and_block.items()
            if tier == "gpu"
        }

    @staticmethod
    def _normalize_tier(tier: str) -> str:
        normalized = tier.lower()
        if normalized not in {"gpu", "cpu"}:
            raise ValueError(f"Unsupported KV cache tier: {tier!r}")
        return normalized

    def get_replicas(self, block_hash: str, tier: str = "gpu") -> set[str] | None:
        """Return replicas for a block in the selected tier."""
        return self.replicas_by_tier_and_block.get(
            (self._normalize_tier(tier), block_hash)
        )

    def clear_replica(self, replica_id: str, tier: str | None = None) -> None:
        """Clear a replica from one tier, or from both tiers by default."""
        tiers = ("gpu", "cpu") if tier is None else (tier,)
        for selected_tier in tiers:
            normalized_tier = self._normalize_tier(selected_tier)
            stale_keys: list[tuple[str, str]] = []
            for key, replicas in self.replicas_by_tier_and_block.items():
                current_tier, _ = key
                if current_tier != normalized_tier:
                    continue
                if replica_id in replicas:
                    replicas.discard(replica_id)
                    if not replicas:
                        stale_keys.append(key)
            for key in stale_keys:
                del self.replicas_by_tier_and_block[key]

    def add_blocks(
        self,
        replica_id: str,
        block_hashes: Iterable[str],
        tier: str = "gpu",
    ) -> None:
        """Add blocks to a tier-specific reverse index."""
        normalized_tier = self._normalize_tier(tier)
        for block_hash in block_hashes:
            self.replicas_by_tier_and_block.setdefault(
                (normalized_tier, block_hash), set()
            ).add(replica_id)

    def remove_blocks(
        self,
        replica_id: str,
        block_hashes: Iterable[str],
        tier: str = "gpu",
    ) -> None:
        """Remove blocks from a tier-specific reverse index."""
        normalized_tier = self._normalize_tier(tier)
        for block_hash in block_hashes:
            key = (normalized_tier, block_hash)
            if key in self.replicas_by_tier_and_block:
                self.replicas_by_tier_and_block[key].discard(replica_id)
                if not self.replicas_by_tier_and_block[key]:
                    del self.replicas_by_tier_and_block[key]
