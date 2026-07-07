"""CollectorProvider — lifecycle manager for data collectors.

Strategies no longer query metrics through the provider — they read from the
unified ``DataStore`` (which wraps the singleton ``MetricsStore`` /
``KVCacheStore``). The provider now owns only collector construction and
lifecycle (start/stop); metric-query proxies that used to live here have moved
to ``DataStore`` (see ``DataStore.get_retained_occupancy``).
"""

from __future__ import annotations

from uni_agent.llm_router.collectors.collector import Collector, get_collector
from uni_agent.llm_router.config.collector import CollectorConfig


class CollectorProvider:
    """Lifecycle manager for data collectors.

    Args:
        collectors_config: ``CollectorConfig`` — connection tuning parameters.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
        server_addresses: ``{node_id: ip:port}`` for HTTP transport.
        kv_event_endpoints: ``{node_id: [sub_addr, replay_addr]}`` for ZMQ transport.
    """

    def __init__(
        self,
        collectors_config: CollectorConfig,
        collection_names: list[str],
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ) -> None:
        self._collectors: list[Collector] = [
            get_collector(
                name,
                collectors_config,
                server_addresses=server_addresses,
                kv_event_endpoints=kv_event_endpoints,
            )
            for name in collection_names
        ]

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Stop all collectors."""
        for collector in self._collectors:
            collector.stop()
