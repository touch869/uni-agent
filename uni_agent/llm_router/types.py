"""Shared contracts for the LLM router strategy layer.

These lightweight types are the strategy module's view of the rest of the
router. They are centralized here so that ``strategy.py`` can be implemented and
unit-tested in isolation, before ``balancer.py`` / ``metrics/provider.py`` exist.

- ``RouteContext`` / ``ReplicaInfo``: per-request context and replica identity.
  The full balancer (future ``balancer.py``) will import/re-export these.
- ``MetricsProvider``: the read-only view a strategy needs. The real metrics
  provider (future ``metrics/provider.py``) satisfies this Protocol structurally.

See ``router_doc/design/detailed_strategy_module.md`` §2.2 for the cross-module
type table.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@runtime_checkable
class MetricsProvider(Protocol):
    """Read-only metrics view consumed by routing strategies.

    A strategy queries runtime data exclusively through these methods; it never
    collects metrics itself. The concrete provider (future
    ``metrics/provider.py``) satisfies this Protocol structurally.
    """

    def get_gpu_utilization(self, replica_id: str) -> float:
        """GPU KV cache usage ratio (0~1). Missing replica returns 0.0."""
        ...

    def get_queue_depth(self, replica_id: str) -> int:
        """Number of requests waiting in the queue. Missing replica returns 0."""
        ...

    def get_gpu_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int]) -> float:
        """GPU prefix cache hit rate (0~1) from the GPU KVC table.

        Returns 0.0 when no ZMQ subscription / no data.
        """
        ...

    def get_tier_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], tier: str) -> float:
        """Per-request multi-tier hit rate from the GPU eviction table.

        ``tier`` is e.g. ``"cpu"`` or ``"ssd"``. Estimates local Mooncake hit for
        blocks evicted from this replica's GPU. Returns 0.0 when unavailable.
        """
        ...
