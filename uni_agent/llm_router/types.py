"""Shared contracts for the LLM router strategy layer.

These lightweight types are the strategy module's view of the rest of the
router. They are centralized here so that ``strategy.py`` can be implemented and
unit-tested in isolation, before ``metrics/provider.py`` exists.

- ``RouteContext`` / ``ReplicaInfo``: per-request context and replica identity.
- ``MetricKey``: canonical metric key constants used with ``RouteDataProvider.get_metric()``.
- ``ReplicaMetrics``: dict-based polled-metrics container (matches ``metric_spec/replica_metrics.py``).
- ``RouteDataProvider``: the read-only view a strategy needs. The real provider
  (``metrics/provider.py``) satisfies this Protocol structurally.

See ``uni_agent/llm_router/docs/design/detailed_strategy_module.md`` §2.2 and
``router_doc/design/metrics模块设计.md`` §4 for the cross-module type table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RouteContext:
    """Context of a single routing request.

    Attributes:
        request_id: Request identifier (used by sticky-session strategies).
        prompt_ids: Token ids of the prompt, used for content-aware routing such
            as prefix cache hit estimation. ``None`` when unavailable.
    """

    request_id: str
    prompt_ids: list[int] | None = None


@dataclass(frozen=True)
class ReplicaInfo:
    """Lightweight identifier of a replica participating in routing.

    Only carries ``replica_id``; the actor handle is held by the balancer and is
    never exposed to strategies.
    """

    replica_id: str


class MetricKey:
    """Canonical metric key constants — backend-agnostic, used with ``RouteDataProvider.get_metric()``.

    Matches ``metric_spec/keys.py::MetricKey`` in the metrics module.
    """

    KV_CACHE_USAGE_PERC = "kv_cache_usage_perc"
    NUM_REQUESTS_RUNNING = "num_requests_running"
    NUM_REQUESTS_WAITING = "num_requests_waiting"


@dataclass
class ReplicaMetrics:
    """Dict-based polled-metrics container for a single replica.

    Matches ``metric_spec/replica_metrics.py::ReplicaMetrics``. Populated by
    ``RouteDataProvider.get_metrics()`` from ``ReplicaMetricsStore``.

    Keys are canonical metric keys (``MetricKey.*``). The ``get()`` method
    returns a safe default when a key is absent.
    """

    replica_id: str
    _data: dict[str, float | int] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: float | int = 0) -> float | int:
        """Query by canonical key. Returns ``default`` when the key is absent."""
        return self._data.get(key, default)

    def set(self, key: str, value: float | int) -> None:
        self._data[key] = value

    def merge(self, other: ReplicaMetrics) -> ReplicaMetrics:
        """Return a new ReplicaMetrics with ``other``'s keys overlaid on this one."""
        merged = dict(self._data)
        merged.update(other._data)
        return ReplicaMetrics(replica_id=self.replica_id, _data=merged)

    def available_keys(self) -> list[str]:
        return list(self._data.keys())


@runtime_checkable
class RouteDataProvider(Protocol):
    """Read-only data view consumed by routing strategies.

    Mirrors ``metrics/provider.py::RouteDataProvider`` (§4.1 of metrics design).
    Strategies query all runtime data through these four methods; they never
    collect metrics themselves.

    Methods:
        get_metric:              Generic canonical-key query from ReplicaMetricsStore.
        get_metrics:             One-shot snapshot of all polled metrics for a replica.
        get_gpu_prefix_hit_rate: Real-time GPU prefix cache hit, delegated to EventCollector.
        get_tier_prefix_hit_rate: Slow-path tier hit rate (v1: snapshot; v2: Mooncake API).
    """

    def get_metric(self, replica_id: str, key: str) -> float | int:
        """Query a single polled metric by canonical key (``MetricKey.*``).

        Delegates to ``ReplicaMetricsStore.get(replica_id).get(key)``.
        Returns the ``METRIC_FIELD_DEFS`` default when the replica or key is unknown.
        """
        ...

    def get_metrics(self, replica_id: str) -> ReplicaMetrics:
        """Return the full polled-metrics snapshot for a replica.

        Avoids multiple ``get_metric`` calls per replica in the hot path.
        Returns a default empty ``ReplicaMetrics`` when the replica is unknown.
        """
        ...

    def get_gpu_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int]) -> float:
        """Real-time GPU prefix cache hit rate (0–1).

        Delegated to ``EventCollector`` → ``KVCacheIndex``.
        Returns 0.0 when no event subscriber is active or no data is available.
        """
        ...

    def get_tier_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], tier: str) -> float:
        """Slow-path per-tier prefix cache hit rate (0–1).

        v1: reads aggregated value from the polling snapshot.
        v2: calls Mooncake /batch_query_keys API.
        ``tier`` is ``"cpu"`` or ``"ssd"``. Returns 0.0 when unavailable.
        """
        ...
