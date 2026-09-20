"""Ownership-aware projection for Router callback state."""

from __future__ import annotations

import threading
from collections import OrderedDict
from enum import Enum
from typing import Any

from .collectors.parse import StickyUpdate
from .store import DataStore


class RouterStateMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    PROJECTOR = "projector"


class RouterStickyStateProjector:
    """Project one bounded sticky-binding view with an explicit commit owner."""

    def __init__(self, mode: RouterStateMode, store: DataStore, *, max_bindings: int = 10_000) -> None:
        if mode is RouterStateMode.LEGACY:
            raise ValueError("legacy mode does not create a Router sticky Projector")
        if max_bindings <= 0:
            raise ValueError("max_bindings must be positive")
        self._mode = mode
        self._store = store
        self._max_bindings = max_bindings
        self._bindings: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()
        self._updates = 0
        self._parity_mismatches = 0
        self._last_error: str | None = None

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._last_error is None

    def apply(self, update: StickyUpdate) -> None:
        """Commit one update when the Projector owns this input family."""
        self._reduce(update, commit=True)

    def observe(self, update: StickyUpdate) -> None:
        """Reduce comparison state after the legacy writer has committed."""
        self._reduce(update, commit=False)

    def binding(self, request_id: str) -> str | None:
        with self._lock:
            return self._bindings.get(request_id)

    def clear(self, *, commit: bool) -> int:
        """Clear sticky bindings through the selected owner and reset comparison state."""
        try:
            with self._lock:
                cleared = self._store.clear_sticky_bindings() if commit else len(self._bindings)
                self._bindings.clear()
                self._updates += 1
                if self._store.sticky_status()["size"] != 0:
                    self._parity_mismatches += 1
                return cleared
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode.value,
                "commit_owner": "projector" if self._mode is RouterStateMode.PROJECTOR else "legacy",
                "healthy": self._last_error is None,
                "last_error": self._last_error,
                "updates": self._updates,
                "binding_count": len(self._bindings),
                "parity_mismatches": self._parity_mismatches,
            }

    def _reduce(self, update: StickyUpdate, *, commit: bool) -> None:
        try:
            touched = self._validate(update)
            with self._lock:
                if commit:
                    self._commit(update)
                self._project(update)
                self._updates += 1
                self._parity_mismatches += sum(
                    self._store.get_sticky_binding(request_id) != self._bindings.get(request_id)
                    for request_id in touched
                )
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    @staticmethod
    def _validate(update: StickyUpdate) -> tuple[str, ...]:
        if update.action == "put":
            if not update.request_id or not update.replica_id:
                raise ValueError("sticky put requires request_id and replica_id")
            return (update.request_id,)
        if update.action == "invalidate":
            if not update.request_id:
                raise ValueError("sticky invalidate requires request_id")
            return (update.request_id,)
        if update.action == "invalidate_replica":
            if not update.replica_ids:
                raise ValueError("sticky replica invalidation requires replica_ids")
            return ()
        raise ValueError(f"unsupported sticky action {update.action!r}")

    def _commit(self, update: StickyUpdate) -> None:
        if update.action == "put":
            self._store.put_sticky_binding(update.request_id, update.replica_id)
        elif update.action == "invalidate":
            self._store.invalidate_sticky_binding(update.request_id)
        else:
            for replica_id in update.replica_ids:
                self._store.invalidate_sticky_replica(replica_id)

    def _project(self, update: StickyUpdate) -> None:
        if update.action == "put":
            self._bindings[update.request_id] = update.replica_id
            self._bindings.move_to_end(update.request_id)
            while len(self._bindings) > self._max_bindings:
                self._bindings.popitem(last=False)
        elif update.action == "invalidate":
            self._bindings.pop(update.request_id, None)
        else:
            removed = set(update.replica_ids)
            self._bindings = OrderedDict(
                (request_id, replica_id)
                for request_id, replica_id in self._bindings.items()
                if replica_id not in removed
            )


__all__ = ["RouterStateMode", "RouterStickyStateProjector"]
