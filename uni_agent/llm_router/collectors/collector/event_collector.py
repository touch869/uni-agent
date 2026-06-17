"""
EventCollector — base class for event-driven data collectors.

Subclasses implement event source-specific logic (ZMQ, HTTP push, etc.)
in ``_subscribe_loop()`` and ``_consume_payload()``.
"""

from __future__ import annotations

import asyncio

from abc import ABC, abstractmethod
from typing import Any


class EventCollector(ABC):
    """Base class for event-driven data collectors.

    Subclasses implement event source-specific logic
    (ZMQ, HTTP push, gRPC stream, etc.).
    """

    def __init__(self) -> None:
        self._store: Any = None
        self._task: asyncio.Task | None = None

    def start(self, store: Any) -> None:
        """Start background event subscription.

        Args:
            store: Data store instance — subclass defines the concrete type.
                   Endpoints are already set from config in ``__init__``.
        """
        self._store = store
        self._task = asyncio.get_running_loop().create_task(self._subscribe_loop())

    def stop(self) -> None:
        """Stop background event subscription."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    @abstractmethod
    async def _subscribe_loop(self) -> None:
        """Background main loop — subclass implements."""
        ...
