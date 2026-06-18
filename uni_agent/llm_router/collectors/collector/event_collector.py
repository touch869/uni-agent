"""
EventCollector — base class for event-driven data collectors.

Subclasses implement event source-specific logic (ZMQ, HTTP push, etc.)
in ``_subscribe_loop()`` and ``_consume_payload()``.

The collector owns its event loop + background thread internally, so
``start()`` / ``stop()`` are plain synchronous calls — callers (provider,
balancer) never need a running asyncio loop.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

from abc import ABC, abstractmethod
from typing import Any

from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("event-collector")

class EventCollector(ABC):
    """Base class for event-driven data collectors.

    Subclasses implement event source-specific logic
    (ZMQ, HTTP push, gRPC stream, etc.). The asyncio event loop and the
    background thread are owned here so callers invoke plain sync
    ``start(store)`` / ``stop()``.
    """

    def __init__(self) -> None:
        self._store: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task: Future | None = None

    def start(self, store: Any) -> None:
        """Start background event subscription (synchronous).

        Spins up a dedicated event loop on a daemon thread and schedules
        ``_subscribe_loop`` on it. Returns once the task is scheduled.

        Args:
            store: Data store instance — subclass defines the concrete type.
                   Endpoints are already set from config in ``__init__``.
        """
        self._store = store
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(self._subscribe_loop(), self._loop)
        logger.info(f"start event loop in background.")

    def stop(self) -> None:
        """Stop background event subscription (synchronous)."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    @abstractmethod
    async def _subscribe_loop(self) -> None:
        """Background main loop — subclass implements."""
        ...
