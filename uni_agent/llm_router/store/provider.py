"""StoreProvider — unified query entry point for routing decisions.

Strategy layers call ``StoreProvider`` methods to read routing data.
It delegates to the singleton store instances — ``MetricsStore`` for
polling metrics and ``KVCacheStore`` for GPU prefix cache data.
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.store.metrics_store import MetricsStore


class StoreProvider:
    """Unified query entry point for routing data.

    Wraps the singleton stores and exposes a single interface for all
    routing queries.  Stateless — instantiate once and reuse.
    """

    def __init__(self) -> None:
        self._metrics = MetricsStore.singleton()
        self._kv = KVCacheStore.singleton()

    # ── Polling metrics ─────────────────────────────────────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a single polling metric by canonical key.

        Args:
            node_id: Target node.
            key: ``MetricKey`` constant, e.g. ``MetricKey.KV_CACHE_USAGE_PERC``.

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if absent.
        """
        return self._metrics.get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full polling metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self._metrics.get(node_id)

    def get_metric_node_ids(self) -> list[str]:
        """Return all node IDs currently in MetricsStore."""
        return self._metrics.all_ids()

    # ── KV cache prefix hit rate ────────────────────────────────────────

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Match prefix hashes against cached blocks, return per-node hit percent.

        Args:
            prompt_ids: Current request's prompt token IDs.

        Returns:
            Dict of node_id → prefix_match_percent (0–100).
            Empty dict if block_size is unknown or no full blocks.
        """
        return self._kv.get_gpu_prefix_hit_rate(prompt_ids)

    def get_tier_prefix_hit_rate(
        self, node_id: str, prompt_ids: list[int], tier: str,
    ) -> float:
        """Query tier-level prefix cache hit rate.

        Args:
            node_id: Target node.
            prompt_ids: Current request's prompt token IDs.
            tier: ``"cpu"`` or ``"ssd"``.

        Returns:
            Hit rate 0.0–1.0.
        """
        return self._kv.get_tier_prefix_hit_rate(node_id, prompt_ids, tier)
