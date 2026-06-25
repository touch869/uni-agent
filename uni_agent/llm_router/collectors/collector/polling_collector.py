"""PollingCollector — base class for Prometheus polling collectors.

Construction accepts a ``CollectorsConfig`` for all configuration.
Each collector creates its own store instance (``store_cls``) in ``__init__``.
Both ``start()`` and ``stop()`` are synchronous — async logic is
encapsulated internally so the upper layer (e.g. Ray actor) can call
them without ``await``.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from abc import ABC, abstractmethod
from concurrent.futures import Future

import httpx

logger = logging.getLogger(__name__)

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig


class PollingCollector(ABC):
    """Base class for Prometheus polling collectors.

    Subclasses implement ``_parse_response()`` with their backend-specific
    parsing logic and declare ``store_cls`` as a class attribute.

    Args:
        config: ``CollectorConfig`` — provides interval, timeout, and
                server_address (list of ``ip:port`` strings).
    """

    def __init__(self, config: CollectorConfig, server_addresses: dict[str, str] | None = None) -> None:
        http_polling = config.http_polling
        self._interval = http_polling["polling_interval"]
        self._http_timeout = http_polling["http_timeout"]

        # replica_id = server_address (ip:port); each address polls its own
        # Prometheus endpoint at ``http://{address}/metrics``
        addresses = server_addresses or {}
        self._endpoints: dict[str, str] = {
            replica_id: f"http://{address}/metrics" for replica_id, address in addresses.items()
        }
        self._client: httpx.AsyncClient | None = None
        self._future: Future | None = None
        # Dedicated event loop for background async work
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background polling.

        Spawns a dedicated event-loop thread, creates the HTTP client
        and the polling task on it.  Synchronous — no ``await`` needed.
        """
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._loop_thread.start()
        # Create AsyncClient on the event-loop thread — it must live on
        # the same loop that uses it.
        client_future = asyncio.run_coroutine_threadsafe(
            self._create_client(), self._loop,
        )
        client_future.result()  # block until client is created
        self._future = asyncio.run_coroutine_threadsafe(
            self._polling_loop(), self._loop,
        )

    async def _create_client(self) -> None:
        """Create the AsyncClient on the event-loop thread."""
        self._client = httpx.AsyncClient(timeout=self._http_timeout)

    def stop(self) -> None:
        """Stop background polling, await task cleanup, and close HTTP client.

        Cancels the polling task, closes the HTTP client on the event-loop
        thread, then stops the loop and joins the thread.  Synchronous —
        blocks until cleanup is complete.
        """
        if self._future is not None:
            self._future.cancel()
            try:
                self._future.result()
            except (asyncio.CancelledError, Exception) as exc:
                logger.debug("Error waiting for polling task to finish: %s", exc)
            self._future = None
        if self._client is not None:
            close_future = asyncio.run_coroutine_threadsafe(
                self._client.aclose(), self._loop,
            )
            try:
                close_future.result(timeout=10)
            except Exception as exc:
                logger.debug("Error closing HTTP client: %s", exc)
            self._client = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

    # ── Background polling loop ─────────────────────────────────────────

    async def _polling_loop(self) -> None:
        """Background loop: poll all endpoints at ``_interval``, parse, write to store."""
        try:
            while True:
                results: dict[str, dict[str, Any]] = {}
                coros = {nid: self._client.get(url) for nid, url in self._endpoints.items()}  # type: ignore[union-attr]
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses):
                    if isinstance(resp, Exception):
                        continue  # failed node — caller falls back to defaults
                    try:
                        results[nid] = self._parse_response(resp.text, nid)  # type: ignore[union-attr]
                    except Exception as exc:
                        # Malformed response — skip this node this round so the
                        # polling loop keeps running instead of breaking entirely.
                        logger.debug("Failed to parse metrics response from node %s: %s", nid, exc)
                        continue
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
