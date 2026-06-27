"""Collector — unified collector interface combining Transport + Decoder.

Composes a ``Transport`` (data acquisition), a ``Decoder`` (data parsing),
and writes results via ``DataStore``.  Both ``start()`` and ``stop()``
are synchronous — async logic is encapsulated internally so the upper layer
(e.g. Ray actor) can call them without ``await``.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from concurrent.futures import Future
from typing import Any

from uni_agent.llm_router.collectors.transport.base import Transport
from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.decoder.vllm.kv_update import KVCacheUpdate
from uni_agent.llm_router.collectors.decoder.vllm.metrics_update import MetricsUpdate
from uni_agent.llm_router.store.data_store import DataStore

logger = logging.getLogger(__name__)


class Collector:
    """Unified collector — composes Transport + Decoder.

    The Collector owns the lifecycle: it starts the Transport on a
    dedicated event-loop thread.  The handler calls Decoder.decode()
    and writes results via DataStore.

    Args:
        transport: Transport instance (ZMQ, HTTP, etc.)
        decoder: Decoder instance (vLLM KV, vLLM Metrics, etc.)
    """

    def __init__(self, transport: Transport, decoder: Decoder) -> None:
        self._transport = transport
        self._decoder = decoder
        self._data_store = DataStore()
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    # ── Derived attributes ─────────────────────────────────────────────

    @property
    def store_cls(self) -> type:
        """Store class — derived from the Decoder."""
        return self._decoder.store_cls

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe.

        Spawns a dedicated event-loop thread, starts the Transport's
        subscribe loop on it.  The handler calls Decoder.decode() and
        writes results via DataStore.  Synchronous — no ``await`` needed.
        """
        def handler(raw_data: bytes | str, node_id: str) -> None:
            """Handler: decode and write via DataStore."""
            result = self._decoder.decode(raw_data, node_id)
            if result is None:
                return

            # Dispatch to appropriate store write method
            if isinstance(result, KVCacheUpdate):
                self._write_kv_update(result)
            elif isinstance(result, MetricsUpdate):
                self._write_metrics_update(result)

        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._loop_thread.start()

        self._future = asyncio.run_coroutine_threadsafe(
            self._transport.subscribe(handler),
            self._loop,
        )

    def _write_kv_update(self, update: KVCacheUpdate) -> None:
        """Write KVCacheUpdate via DataStore."""
        # Learn block_size from first event
        if update.block_size is not None:
            self._data_store.set_block_size(update.block_size)

        # Apply operations
        if update.clear_all:
            self._data_store.clear_kv_node(update.node_id)
        if update.remove_blocks:
            self._data_store.remove_kv_blocks(update.node_id, update.remove_blocks)
        if update.add_blocks:
            self._data_store.add_kv_blocks(update.node_id, update.add_blocks)

    def _write_metrics_update(self, update: MetricsUpdate) -> None:
        """Write MetricsUpdate via DataStore."""
        self._data_store.refresh_metrics({update.node_id: update.metrics})

    def stop(self) -> None:
        """Stop the collector — stop transport, then stop event-loop thread.

        Synchronous — blocks until all cleanup is complete.
        """
        # First stop the transport (it cancels its own tasks and closes sockets)
        self._transport.stop()

        # Cancel and await the main subscribe future
        if self._future is not None:
            self._future.cancel()
            try:
                self._future.result()
            except (asyncio.CancelledError, Exception) as exc:
                logger.debug("Error waiting for collector task to finish: %s", exc)
            self._future = None

        # Stop the event loop and join the thread
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None


# ── Factory function ───────────────────────────────────────────────────


def get_collector(name: str, **kwargs: Any) -> Collector:
    """Create a Collector by name.

    Args:
        name: Collector type — "vllm_metrics" or "vllm_zmq".
        **kwargs: Transport constructor arguments.

    Returns:
        Configured Collector instance.

    Raises:
        ValueError: If name is unknown.

    Examples:
        # Metrics collector (HTTP polling)
        collector = get_collector(
            "vllm_metrics",
            endpoints={"node1": "127.0.0.1:8000"},
            interval=5.0,
            http_timeout=10.0
        )

        # KV cache collector (ZMQ events)
        collector = get_collector(
            "vllm_zmq",
            endpoints={"node1": "127.0.0.1:5555"}
        )
    """
    if name == "vllm_metrics":
        from uni_agent.llm_router.collectors.transport.http import HTTPTransport
        from uni_agent.llm_router.collectors.decoder.vllm.metrics import VLLMMetricsDecoder

        transport = HTTPTransport(**kwargs)
        decoder = VLLMMetricsDecoder()
        return Collector(transport, decoder)

    elif name == "vllm_zmq":
        from uni_agent.llm_router.collectors.transport.zmq import ZMQTransport
        from uni_agent.llm_router.collectors.decoder.vllm.kv import VLLMKVDecoder

        transport = ZMQTransport(**kwargs)
        decoder = VLLMKVDecoder()
        return Collector(transport, decoder)

    else:
        raise ValueError(
            f"Unknown collector: '{name}'. "
            f"Available: ['vllm_metrics', 'vllm_zmq']"
        )
