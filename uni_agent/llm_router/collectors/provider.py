"""RouteDataProvider — unified query entry point for routing decisions.

Strategy layers call ``RouteDataProvider`` methods to get metrics data.
It delegates to store instances (``MetricsStore`` for polling metrics,
``KVCacheStore`` for GPU prefix cache data).

Each collector creates its own store instance in ``__init__`` via
``store_cls()``.  Stores are singletons — calling ``MetricsStore()``
or ``KVCacheStore()`` always returns the shared instance, so dedup
happens automatically at the store level, not in the provider.

All query computations are delegated to the respective store classes.
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.collector.polling_collector import PollingCollector
from uni_agent.llm_router.collectors.collector.event_collector import EventCollector
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore
from uni_agent.llm_router.collectors.registry import BUILTIN_REGISTRY


class RouteDataProvider:
    """Unified query entry point — strategies use this to access all metrics.

    ``RouteDataProvider`` creates collectors.  Store deduplication is
    handled by the store classes themselves (singleton pattern) — no
    manual dedup needed here.

    All query computations are delegated to the respective store classes.

    Args:
        collectors_config: ``CollectorConfig`` — provides common settings
            and endpoint addresses.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
    """

    def __init__(
        self,
        collectors_config,
        collection_names,
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ) -> None:
        self._collectors: list[Any] = []
        for name in collection_names:
            collector_cls = BUILTIN_REGISTRY.get_collector(name)
            if issubclass(collector_cls, PollingCollector):
                self._collectors.append(
                    collector_cls(config=collectors_config, server_addresses=server_addresses)
                )
            elif issubclass(collector_cls, EventCollector):
                self._collectors.append(
                    collector_cls(config=collectors_config, kv_event_addresses=kv_event_endpoints)
                )
            else:
                raise TypeError(f"unsupport collector: {collector_cls}.")

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Stop all collectors and await their cleanup."""
        for collector in self._collectors:
            collector.stop()

    # ── Convenience accessors for commonly-used stores ────────────────

    @property
    def _metrics_store(self) -> MetricsStore:
        """Get the shared MetricsStore singleton."""
        return MetricsStore.default()

    @property
    def _kv_store(self) -> KVCacheStore:
        """Get the shared KVCacheStore singleton."""
        return KVCacheStore.default()

    # ── Generic polling metric queries (canonical key) ──────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a polling metric by canonical key.

        Delegates to ``MetricsStore.get(node_id, key)``.

        Args:
            node_id: Target node.
            key: ``MetricKey`` constant, e.g. ``MetricKey.KV_CACHE_USAGE_PERC``.

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if
            node or metric is absent.
        """
        return self._metrics_store.get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full polling metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node
            is absent in the store.
        """
        return self._metrics_store.get(node_id)

    # ── KVCache prefix hit rate queries ────────────────────────────────

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Match prefix hashes against cached blocks, return per-replica hit percent.

        Delegates to ``KVCacheStore.get_gpu_prefix_hit_rate(prompt_ids)``.
        """
        return self._kv_store.get_gpu_prefix_hit_rate(prompt_ids)

    def get_tier_prefix_hit_rate(
        self, node_id: str, prompt_ids: list[int], tier: str,
    ) -> float:
        """Query tier-level prefix cache hit rate (slow-path data).

        Delegates to ``KVCacheStore.get_tier_prefix_hit_rate``.
        """
        return self._kv_store.get_tier_prefix_hit_rate(node_id, prompt_ids, tier)
