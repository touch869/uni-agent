"""Registry — lazy registry for collector classes.

Built-in entries are registered as ``collection_name → "module.path:ClassName"``
strings.  The store class is derived from the collector's ``store_cls``
attribute — no separate store registration needed.
"""

from __future__ import annotations

import importlib


class Registry:
    """Lazy registry — maps collection_name to collector class.

    ``register_entry(name, collector_path)`` stores a
    ``"module.path:ClassName"`` string; the class is imported on first
    ``get_collector`` call and cached in-place.

    ``get_store(name)`` resolves the collector class first, then reads
    its ``store_cls`` attribute — each collector declares its own store.
    """

    def __init__(self) -> None:
        self._entries: dict[str, type | str] = {}

    def register_entry(self, name: str, collector_path: str) -> None:
        """Register a lazy entry-point path for a collector.

        Paths use ``"module.path:ClassName"`` format (setuptools entry-point style).
        """
        self._entries[name] = collector_path

    def get_collector(self, name: str) -> type:
        """Look up a collector class.  Resolves lazy entries on first call."""
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown collector: '{name}'. Registered: {sorted(self._entries.keys())}"
            )
        if isinstance(entry, type):
            return entry
        # Lazy import — "module.path:ClassName"
        module_path, class_name = entry.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        self._entries[name] = cls  # cache-in-place: string → class
        return cls

    def get_store(self, name: str) -> type:
        """Look up the store class for a collector.

        Derived from the collector's ``store_cls`` class attribute —
        each collector declares its own store, so no separate registration
        is needed.
        """
        collector_cls = self.get_collector(name)
        return collector_cls.store_cls


# ── Module-level built-in registration (lazy) ─────────────────────────

BUILTIN_REGISTRY = Registry()
BUILTIN_REGISTRY.register_entry(
    "vllm_metrics",
    "uni_agent.llm_router.collectors.collector.vllm.polling_collector:VLLMPollingCollector",
)
BUILTIN_REGISTRY.register_entry(
    "vllm_zmq",
    "uni_agent.llm_router.collectors.collector.vllm.event_collector:VLLMKVEventCollector",
)
