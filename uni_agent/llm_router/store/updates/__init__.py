"""Structured update commands — decoder output, consumed by DataStore.apply().

Each Update is a pure dataclass produced by a Decoder and applied to the
appropriate singleton store via ``DataStore.apply(update)``.  They live
under ``store/`` (not ``collectors/``) so the dependency direction stays
one-way: ``collectors → store``, with no ``store → collectors`` back-edge.
"""

from uni_agent.llm_router.store.updates.kv_update import KVCacheUpdate
from uni_agent.llm_router.store.updates.metrics_update import MetricsUpdate

__all__ = ["KVCacheUpdate", "MetricsUpdate"]
