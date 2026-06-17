"""PollingCollector — base class for Prometheus polling collectors.

Construction accepts a ``CollectorsConfig`` for all configuration.
``start(store)`` must be called to begin background polling.
``stop()`` cancels the background task and closes the HTTP client.
"""

from __future__ import annotations

import asyncio

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
        config: ``CollectorConfig`` — provides interval, timeout, and
                server_address (list of ``ip:port`` strings).
    """

    def __init__(self, config) -> None:
        self._interval = config.retry_interval
        self._http_timeout = config.timeout
        # replica_id = server_address (ip:port); each address polls its own
        # Prometheus endpoint at ``http://{address}/metrics``
        self._endpoints: dict[str, str] = {
            addr: f"http://{addr}/metrics" for addr in config.server_address
        }
        self._store: MetricsStore | None = None
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task | None = None

    def start(self, store: MetricsStore) -> None:
        """Start background polling.

        Args:
            store: ``MetricsStore`` — polling results are written via
                   ``store.refresh()``.
        """
        self._store = store
        self._client = httpx.AsyncClient(timeout=self._http_timeout)
        self._task = asyncio.get_running_loop().create_task(self._polling_loop())

    def stop(self) -> None:
        """Stop background polling and close HTTP client."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._client is not None:
            try:
                asyncio.get_running_loop().create_task(self._client.aclose())
            except RuntimeError:
                pass
            self._client = None

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
