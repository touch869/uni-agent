"""PollingCollector — base class for Prometheus polling collectors.

Construction accepts a ``CollectorsConfig`` for all configuration.
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


class PollingCollector(ABC):
    """Base class for Prometheus polling collectors.

    Subclasses implement ``_parse_response()`` with their backend-specific
    parsing logic.

    Args:
        config: ``CollectorConfig`` — http_polling params
                (polling_interval / http_timeout).
    """

    def __init__(self, config) -> None:
        http_polling = config.http_polling
        self._interval = http_polling["polling_interval"]
        self._http_timeout = http_polling["http_timeout"]
        # TODO(server_address): the per-replica metrics addresses (ip:port) are
        # allocated dynamically when the vLLM servers start and passed down at
        # runtime — they must NOT live in the static CollectorConfig.
        # Hardcoded placeholder for bring-up; real injection is a collectors-
        # module design item.
        server_address = ["127.0.0.1:8000"]
        # replica_id = server_address (ip:port); each address polls its own
        # Prometheus endpoint at ``http://{address}/metrics``
        self._endpoints: dict[str, str] = {
            addr: f"http://{addr}/metrics" for addr in server_address
        }
        self._store: MetricsStore | None = None
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task: Future | None = None

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
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(self._polling_loop(), self._loop)

    def stop(self) -> None:
        """Stop background polling and close HTTP client (synchronous).

        Cancels the polling task *inside* the background loop so that
        ``CancelledError`` is properly caught and the coroutine finishes
        cleanly — this avoids ``Task was destroyed but it is pending!``
        warnings.
        """
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
        try:
            while True:
                results: dict[str, dict[str, Any]] = {}
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}  # type: ignore[union-attr]
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses):
                    if isinstance(resp, Exception):
                        continue  # failed node — caller falls back to defaults
                    results[nid] = self._parse_response(resp.text, nid)  # type: ignore[union-attr]
                if self._store is not None:
                    self._store.refresh(results)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    # ── Parsing (abstract — subclass implements) ────────────────────────

    @abstractmethod
    def _parse_response(self, text: str, node_id: str) -> dict[str, Any]:
        """Parse Prometheus exposition-format text into a metrics dict.

        Subclasses inject their own backend mapping and
        implement any backend-specific parsing logic here.
        """
        ...
