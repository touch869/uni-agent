"""PollingCollector — base class for Prometheus polling collectors.

Construction accepts a ``CollectorsConfig`` for all configuration and
a ``server_addresses`` dict for per-replica endpoint discovery.
``start(store)`` must be called to begin background polling.
``stop()`` cancels the background task and closes the HTTP client.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

from abc import ABC, abstractmethod

import httpx

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.store.metrics_store import MetricsStore
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("polling-collector")


class PollingCollector(ABC):
    """Base class for Prometheus polling collectors.

    Subclasses implement ``_parse_response()`` with their backend-specific
    parsing logic.

    Args:
        config: ``CollectorConfig`` — http_polling params
                (polling_interval / http_timeout).
        server_addresses: Per-replica metrics addresses injected by the
                balancer at runtime. Key = replica_id, value = server_address
                (ip:port). Each address polls its own Prometheus endpoint
                at ``http://{address}/metrics``.
    """

    def __init__(self, config, server_addresses: dict[str, str] | None = None) -> None:
        http_polling = config.http_polling
        self._interval = http_polling["polling_interval"]
        self._http_timeout = http_polling["http_timeout"]
        # Per-replica metrics addresses injected by balancer at runtime.
        # Key = replica_id, value = server_address (ip:port).
        addresses = server_addresses or {}
        self._endpoints: dict[str, str] = {
            replica_id: f"http://{address}/metrics"
            for replica_id, address in addresses.items()
        }
        self._store: MetricsStore | None = None
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task: Future | None = None

        logger.info(f"PollingCollector _endpoints: {self._endpoints}")

    def start(self, store: MetricsStore) -> None:
        """Start background polling (synchronous).

        Spins up a dedicated event loop on a daemon thread and schedules
        ``_polling_loop`` on it. The httpx client is created inside the loop
        thread (in ``_polling_loop``) so it binds to the right loop.

        Args:
            store: ``MetricsStore`` — polling results are written via
                   ``store.refresh()``.
        """
        self._store = store
        logger.info(f"starting polling loop, endpoints={self._endpoints}, interval={self._interval}s")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(self._polling_loop(), self._loop)
        logger.info("polling loop started in background thread")

    def stop(self) -> None:
        """Stop background polling and close HTTP client (synchronous).

        Cancels the polling task *inside* the background loop so that
        ``CancelledError`` is properly caught and the coroutine finishes
        cleanly — this avoids ``Task was destroyed but it is pending!``
        warnings.
        """
        logger.info("stopping polling loop")
        if self._task is not None and self._loop is not None:
            # Cancel the task *inside* the loop thread and await its
            # cleanup so CancelledError is consumed before we stop the loop.
            async def _cancel_and_wait():
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            fut = asyncio.run_coroutine_threadsafe(_cancel_and_wait(), self._loop)
            fut.result(timeout=3)
            self._task = None
            logger.info("polling task cancelled and cleaned up")
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        # httpx client lived in the (now-stopped) background loop thread; let it
        # be GC'd rather than closing cross-thread.
        self._client = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # ── Background polling loop ─────────────────────────────────────────

    async def _polling_loop(self) -> None:
        """Background loop: poll all endpoints at ``_interval``, parse, write to store."""
        # Create the httpx client here so it binds to THIS loop (background thread).
        self._client = httpx.AsyncClient(timeout=self._http_timeout)
        logger.debug(f"httpx client created, timeout={self._http_timeout}")
        try:
            while True:
                results: dict[str, dict[str, Any]] = {}
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}  # type: ignore[union-attr]
                logger.debug(f"polling {len(coros)} endpoints: {list(coros.keys())}")
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses):
                    if isinstance(resp, Exception):
                        logger.warning(f"polling {nid} failed: {type(resp).__name__}: {resp}")
                        continue  # failed node — caller falls back to defaults
                    results[nid] = self._parse_response(resp.text, nid)  # type: ignore[union-attr]
                    logger.debug(f"polling {nid} succeeded, parsed metrics keys: {list(results[nid].keys())}")
                if self._store is not None:
                    self._store.refresh(results)
                    logger.debug(f"store refreshed with {len(results)} replica results")
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("polling loop cancelled, exiting")
            pass

    # ── Parsing (abstract — subclass implements) ────────────────────────

    @abstractmethod
    def _parse_response(self, text: str, node_id: str) -> dict[str, Any]:
        """Parse Prometheus exposition-format text into a metrics dict.

        Subclasses inject their own backend mapping and
        implement any backend-specific parsing logic here.
        """
        ...
