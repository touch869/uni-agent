"""RouteDataProvider — unified query entry point for routing decisions.

Strategy layers call ``RouteDataProvider`` methods to get metrics data.
It delegates to store instances (``MetricsStore`` for polling metrics,
``KVCacheStore`` for GPU prefix cache data).

Collectors are created from ``BUILTIN_REGISTRY`` as ``Collector``
instances combining Transport + Decoder.  Stores are singletons —
deduplication happens automatically at the store level.

All query computations are delegated to the respective store classes.
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.store.metrics_store import MetricsStore
from uni_agent.llm_router.collectors.registry import BUILTIN_REGISTRY


class RouteDataProvider:
    """Unified query entry point — strategies use this to access all metrics.

    ``RouteDataProvider`` creates collectors via the registry, which
    combines Transport + Decoder into ``Collector`` instances.  Store
    deduplication is handled by the store classes themselves (singleton).

    All query computations are delegated to the respective store classes.

    Args:
        collectors_config: ``CollectorConfig`` — provides common settings
            and endpoint addresses.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
        server_addresses: ``{replica_id: ip:port}`` for HTTP transport.
        kv_event_endpoints: ``{replica_id: [sub_addr, replay_addr]}`` for ZMQ transport.
    """

    def __init__(
        self,
        collectors_config,
        collection_names,
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ) -> None:
        self._collectors: list[Any] = []

        http_polling = collectors_config.http_polling
        long_conn = collectors_config.long_connection

        for name in collection_names:
            if name == "vllm_metrics":
                collector = BUILTIN_REGISTRY.get_collector(
                    name,
                    endpoints=server_addresses or {},
                    interval=http_polling["polling_interval"],
                    http_timeout=http_polling["http_timeout"],
                )
            elif name == "vllm_zmq":
                collector = BUILTIN_REGISTRY.get_collector(
                    name,
                    endpoints=kv_event_endpoints or {},
                    base_retry_delay=long_conn["base_retry_delay"],
                    max_retry_delay=long_conn["max_retry_delay"],
                    max_retry_attempts=long_conn["max_retry_attempts"],
                    retry_backoff_factor=long_conn["retry_backoff_factor"],
                )
            else:
                collector = BUILTIN_REGISTRY.get_collector(name)
            self._collectors.append(collector)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Stop all collectors and await their cleanup."""
        for collector in self._collectors:
            collector.stop()
