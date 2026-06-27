"""CollectorProvider — lifecycle manager for data collectors.

Creates and manages ``Collector`` instances (Transport + Decoder) via
``get_collector()``.  Stores are singletons managed separately —
use ``DataStore`` to query routing data.
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.collector import get_collector


class CollectorProvider:
    """Lifecycle manager for data collectors.

    Creates collectors via ``get_collector()``, combining Transport + Decoder
    into ``Collector`` instances.  Call ``start()`` / ``stop()`` to
    control the background collection loops.

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
        self._collectors: list[Any] = []

        http_polling = collectors_config.http_polling
        long_conn = collectors_config.long_connection

        for name in collection_names:
            if name == "vllm_metrics":
                collector = get_collector(
                    name,
                    endpoints=server_addresses or {},
                    interval=http_polling["polling_interval"],
                    http_timeout=http_polling["http_timeout"],
                )
            elif name == "vllm_zmq":
                collector = get_collector(
                    name,
                    endpoints=kv_event_endpoints or {},
                    base_retry_delay=long_conn["base_retry_delay"],
                    max_retry_delay=long_conn["max_retry_delay"],
                    max_retry_attempts=long_conn["max_retry_attempts"],
                    retry_backoff_factor=long_conn["retry_backoff_factor"],
                )
            else:
                collector = get_collector(name)
            self._collectors.append(collector)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Stop all collectors."""
        for collector in self._collectors:
            collector.stop()

