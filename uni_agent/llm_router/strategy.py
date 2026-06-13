"""LLM router strategy layer: interface, registry, composition, builtin strategy.

Implements ``router_doc/design/detailed_strategy_module.md``:

- ``RoutingStrategy``  — scoring protocol (§2.1).
- ``StrategyRegistry`` — name → strategy class registry (§2.5).
- ``route()``          — weighted-sum composition + ranking (§4).
- ``KVCacheAwareStrategy`` — single strategy encapsulating load filtering,
  overload handling and fast/slow prefix-cache scoring (§6).

The composition framework is a pure weighted sum: each strategy scores replicas
independently, ``route()`` sums ``weight * score`` and ranks by total. All KVC
business logic (load, overload, fast/slow paths) lives inside the strategy.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uni_agent.llm_router.types import MetricsProvider, ReplicaInfo, RouteContext

logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Strategy construction or scoring error."""


@runtime_checkable
class RoutingStrategy(Protocol):
    """Routing scoring strategy.

    Each strategy scores a batch of replicas independently and returns a list of
    the same length and order. ``route()`` weighted-sums the strategies' outputs.
    """

    def score(
        self,
        ctx: "RouteContext",
        metrics: "MetricsProvider",
        replicas: list["ReplicaInfo"],
    ) -> list[float]:
        """Score each replica. Larger is better; negatives are allowed."""
        ...


class StrategyRegistry:
    """Strategy registry. Decorator registration + lookup by name."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator factory: register a strategy class.

        Re-registering the *same* class under the same name is idempotent (so a
        module re-import does not crash); registering a *different* class under
        an existing name raises ValueError.
        """

        def wrapper(strategy_cls: type) -> type:
            existing = cls._registry.get(name)
            if existing is not None and existing is not strategy_cls:
                raise ValueError(
                    f"Strategy '{name}' already registered. Existing: {existing}"
                )
            cls._registry[name] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get(cls, name: str) -> type:
        """Look up a strategy class by name. Unknown -> ValueError."""
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy: '{name}'. Available: {cls.list_strategies()}")
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return all registered strategy names (sorted)."""
        return sorted(cls._registry.keys())


def _rank_key(score: float) -> float:
    """Sort key that treats non-finite scores (NaN/inf) as worst.

    Guards against a metric returning NaN: a bare comparison sort would place
    NaN at an input-order-dependent position and could rank it first.
    """
    return score if math.isfinite(score) else float("-inf")


def route(
    strategies: list[tuple["RoutingStrategy", float]],
    ctx: "RouteContext",
    metrics: "MetricsProvider",
    replicas: list["ReplicaInfo"],
) -> list[int]:
    """Score each strategy independently, weighted-sum, return replica index ranking.

    Args:
        strategies: ``(strategy, weight)`` pairs; weights normally sum to 1. The
            tuple weight is the single source of truth for composition (a
            strategy's own ``weight`` attribute, if any, is not applied here).
        ctx: routing request context.
        metrics: read-only metrics view.
        replicas: replicas participating in this route.

    Returns:
        Replica indices sorted by total score, best first. Non-finite totals are
        ranked last.

    Raises:
        RuntimeError: ``replicas`` is empty.
        StrategyError: a strategy raises, or returns a list whose length
            != len(replicas) (wrapped with the strategy's class name).
    """
    n = len(replicas)
    if n == 0:
        raise RuntimeError("no available replicas")

    final = [0.0] * n
    for strategy, weight in strategies:
        name = type(strategy).__name__
        try:
            scores = strategy.score(ctx, metrics, replicas)
        except StrategyError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raise with strategy context
            raise StrategyError(f"{name}.score() raised {type(exc).__name__}: {exc}") from exc
        if len(scores) != n:
            raise StrategyError(f"{name}.score() returned {len(scores)} scores, expected {n}")
        for i in range(n):
            final[i] += weight * scores[i]

    return sorted(range(n), key=lambda i: _rank_key(final[i]), reverse=True)


_DEFAULT_LAYER_WEIGHTS = {"cpu": 1.0, "ssd": 0.25}


@StrategyRegistry.register("kv_cache_aware")
class KVCacheAwareStrategy:
    """KVC-aware strategy: load filtering + fast/slow prefix-cache scoring (§6).

    Combined score per replica:
      - all replicas overloaded:    ``alpha * S_cache``    (load dropped, §6.5)
      - overloaded (partial case):  ``S_load`` (< 0)        (sinks below healthy)
      - healthy replica:            ``alpha * S_cache + (1 - alpha) * S_load``

    where ``S_load = 1 - load / load_threshold`` (load = gpu_util * queue_depth)
    and ``S_cache`` is the GPU prefix hit (fast path) or the truncated multi-tier
    weighted hit from the GPU eviction table (slow path).

    The partial-overload case uses ``S_load`` directly (not ``(1-alpha)*S_load``)
    so an overloaded replica is *strictly negative* and ranks below every healthy
    replica (which scores >= 0) regardless of ``alpha`` — including ``alpha == 1``.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        load_threshold: float = 80.0,
        layer_weights: dict[str, float] | None = None,
        weight: float = 1.0,
    ):
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if load_threshold <= 0:
            raise StrategyError(f"load_threshold must be > 0, got {load_threshold}")
        if not 0 < weight <= 1:
            raise StrategyError(f"weight must be in (0, 1], got {weight}")
        # None -> defaults; an explicit (possibly empty) dict is honored as-is.
        self.layer_weights = dict(_DEFAULT_LAYER_WEIGHTS) if layer_weights is None else dict(layer_weights)
        _valid_tiers = {"cpu", "ssd"}
        for tier, w in self.layer_weights.items():
            if tier not in _valid_tiers:
                raise StrategyError(f"layer_weights key must be in {_valid_tiers}, got '{tier}'")
            if w < 0:
                raise StrategyError(f"layer_weights[{tier}] must be >= 0, got {w}")
        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        # Framework-level weight; also passed as the route() tuple weight.
        # route() applies the tuple weight — this attribute is the same value,
        # kept for introspection, and is NOT re-applied here.
        self.weight = float(weight)

    @classmethod
    def from_config(cls, cfg: "KVCAwareStrategyConfig") -> "KVCacheAwareStrategy":
        """Construct from a KVCAwareStrategyConfig (called by Balancer on init)."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=dict(cfg.layer_weights),
            weight=cfg.weight,
        )

    def score(
        self,
        ctx: "RouteContext",
        metrics: "MetricsProvider",
        replicas: list["ReplicaInfo"],
    ) -> list[float]:
        prompt_ids = ctx.prompt_ids or []

        # 1. load & overload (S_load < 0 means load > threshold)
        s_loads: list[float] = []
        overloaded: list[bool] = []
        for r in replicas:
            load = metrics.get_gpu_utilization(r.replica_id) * metrics.get_queue_depth(r.replica_id)
            s = 1.0 - load / self.load_threshold
            s_loads.append(s)
            overloaded.append(s < 0)
        all_overloaded = bool(replicas) and all(overloaded)

        # 2. GPU prefix hit rate, computed once per replica (reused below).
        gpu_hits = [metrics.get_gpu_prefix_hit_rate(r.replica_id, prompt_ids) for r in replicas]

        # fast/slow path: only the healthy subset decides; all-overloaded forces slow.
        use_fast = (not all_overloaded) and any(
            gpu_hits[i] > 0 for i in range(len(replicas)) if not overloaded[i]
        )

        def cache_score(idx: int, r: "ReplicaInfo") -> float:
            if use_fast:
                return gpu_hits[idx]
            return min(1.0, self._slow_cache(metrics, r, prompt_ids))

        # 3. combined scoring
        result: list[float] = []
        slow_cache: list[float] = []  # populated lazily for the warning below
        for i, r in enumerate(replicas):
            if all_overloaded:
                sc = min(1.0, self._slow_cache(metrics, r, prompt_ids))
                slow_cache.append(sc)
                result.append(self.alpha * sc)  # pure cache, §6.5
            elif overloaded[i]:
                result.append(s_loads[i])  # strictly negative -> sinks below healthy
            else:
                result.append(self.alpha * cache_score(i, r) + (1 - self.alpha) * s_loads[i])

        # §7: warn when a fully-overloaded pool has no cache signal to rank by.
        if all_overloaded and not any(gpu_hits) and not any(slow_cache):
            logger.warning(
                "All %d replicas overloaded with no cache signal; routing is undifferentiated.",
                len(replicas),
            )
        return result

    def _slow_cache(self, metrics: "MetricsProvider", r: "ReplicaInfo", prompt_ids: list[int]) -> float:
        """Multi-tier weighted hit rate from the GPU eviction table (slow path)."""
        return sum(
            metrics.get_tier_prefix_hit_rate(r.replica_id, prompt_ids, tier) * w
            for tier, w in self.layer_weights.items()
        )
