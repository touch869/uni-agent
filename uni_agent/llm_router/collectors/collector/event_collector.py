"""
EventCollector — base class for event-driven data collectors.

Subclasses implement event source-specific logic (ZMQ, HTTP push, etc.)
in ``_subscribe_loop()`` and ``_consume_payload()``.  Each collector
creates its own store instance (``store_cls``) in ``__init__`` —
``start()`` no longer takes a store argument.

Both ``start()`` and ``stop()`` are synchronous — they encapsulate
the async logic internally so the upper layer (e.g. Ray actor) can
call them without ``await``.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger(__name__)


class EventCollector(ABC):
    """Base class for event-driven data collectors.

    Subclasses implement event source-specific logic
    (ZMQ, HTTP push, gRPC stream, etc.) and declare ``store_cls``
    as a class attribute.
    """

    def __init__(self) -> None:
        self._future: Future | None = None
        # Dedicated event loop for background async work — runs on its own
        # thread so callers (e.g. Ray actor methods) don't need to be in
        # an async context.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background event subscription.

        Spawns a dedicated event-loop thread and creates the background
        task on it.  Synchronous — no ``await`` needed from the caller.
        """
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._loop_thread.start()
        self._future = asyncio.run_coroutine_threadsafe(
            self._subscribe_loop(), self._loop,
        )

    def stop(self) -> None:
        """Stop background event subscription and clean up.

        Cancels the task, awaits its cleanup on the event-loop thread,
        then stops the loop and joins the thread.  Synchronous — blocks
        until cleanup is complete.
        """
        if self._future is not None:
            self._future.cancel()
            # Block until the cancelled task finishes on the event-loop thread
            try:
                self._future.result()
            except (asyncio.CancelledError, Exception) as exc:
                logger.debug("Error waiting for event task to finish: %s", exc)
            self._future = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

    @abstractmethod
    async def _subscribe_loop(self) -> None:
        """Background main loop — subclass implements."""
        ...
