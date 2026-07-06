"""KVCache-aware runtime strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.strategies.load_score import DEFAULT_LOAD_WEIGHTS, load_normalized
from uni_agent.llm_router.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from uni_agent.llm_router.collectors.provider import RouteDataProvider
    from uni_agent.llm_router.strategies.base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

STICKY_TOP_SCORE = 1e9


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[str, float],
        collector_names: list[str],
        weight: float,
        load_weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_tiers = {"gpu", "cpu", "ssd"}
        if set(layer_weights.keys()) != _valid_tiers:
            raise StrategyError(f"layer_weights keys must be {_valid_tiers}, got {set(layer_weights.keys())}")
        for tier, tier_weight in layer_weights.items():
            if tier_weight < 0:
                raise StrategyError(f"layer_weights[{tier}] must be >= 0, got {tier_weight}")
        weights_sum = sum(layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise StrategyError(f"layer_weights values must sum to 1.0, got {weights_sum}")
        if len(load_weights) != 3 or any(w < 0 for w in load_weights):
            raise StrategyError(f"load_weights must be 3 non-negative values, got {load_weights}")
        if abs(sum(load_weights) - 1.0) > 1e-6:
            raise StrategyError(f"load_weights must sum to 1.0, got {sum(load_weights)}")

        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        self.load_weights = tuple(load_weights)
        self._max_num_seqs: int | None = None
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, load_weights={self.load_weights}"
        )

    def set_capacity(self, max_num_seqs: int) -> None:
        """Inject ``--max-num-seqs`` from the server handle's rollout config."""
        if not isinstance(max_num_seqs, int) or max_num_seqs <= 0:
            raise StrategyError(f"max_num_seqs must be a positive int, got {max_num_seqs}")
        self._max_num_seqs = max_num_seqs
        logger.info(f"KVCacheAwareStrategy capacity set: max_num_seqs={max_num_seqs}")

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> KVCacheAwareStrategy:
        """Construct from config. ``max_num_seqs`` is injected by the Balancer
        via ``set_capacity`` after fetching from the server handle."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
        )

    def _compute_load(self, kv_usage: float, running: int | float, waiting: int | float) -> float:
        if self._max_num_seqs is None:
            raise StrategyError("set_capacity() must be called before routing")
        return load_normalized(kv_usage, running, waiting, max_num_seqs=self._max_num_seqs, weights=self.load_weights)

    def is_overloaded(self, provider: RouteDataProvider, replica: ReplicaInfo) -> bool:
        """Return True if ``replica`` is overloaded (``load > load_threshold``)."""
        m = provider.get_metrics(replica.replica_id)
        kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
        running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
        waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
        return self._compute_load(kv_usage, running, waiting) > self.load_threshold

    def _sticky_shortcut(self, provider, replicas, request_id, sticky_table):
        if not request_id or sticky_table is None:
            return None
        sticky_id = sticky_table.get(request_id)
        if sticky_id is None:
            return None
        for idx, replica in enumerate(replicas):
            if replica.replica_id == sticky_id:
                m = provider.get_metrics(replica.replica_id)
                kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
                running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
                waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
                load = self._compute_load(kv_usage, running, waiting)
                if load > self.load_threshold:
                    logger.info(f"score(): STICKY replica={sticky_id} OVERLOADED → fallback to COMBINED scoring")
                    return None
                logger.info(f"score(): STICKY replica={sticky_id} HIT → short-circuit (top score)")
                scores = [0.0] * len(replicas)
                scores[idx] = STICKY_TOP_SCORE
                return scores
        logger.info(f"score(): sticky replica={sticky_id} not in pool, fallback to combined scoring")
        return None

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        """Score each replica. Larger is better.

        S = α·S_cache + (1-α)·S_load
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []
        shortcut = self._sticky_shortcut(provider, replicas, request_id, sticky_table)
        if shortcut is not None:
            return shortcut
        effective_prompt_ids = prompt_ids or []
        gpu_hit_pct = provider.get_gpu_prefix_hit_rate(effective_prompt_ids)
        result = []
        for replica in replicas:
            m = provider.get_metrics(replica.replica_id)
            kv_usage = m.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            load = self._compute_load(kv_usage, running, waiting)
            s_load = 1.0 - load
            s_cache = self._cache_score(provider, replica, effective_prompt_ids, gpu_hit_pct)
            score = self.alpha * s_cache + (1 - self.alpha) * s_load
            result.append(score)
            gpu_hit = gpu_hit_pct.get(replica.replica_id, 0) / 100.0
            logger.info(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} "
                f"→ load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f} s_cache={s_cache:.4f} "
                f"({self.alpha:.2f}·cache + {1 - self.alpha:.2f}·load) → score={score:.4f}"
            )
        scores_str = ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas))
        logger.info(f"score(): COMBINED scores: {scores_str}")
        return result

    def _cache_score(self, provider, replica, prompt_ids, gpu_hit_pct):
        gpu_hit = gpu_hit_pct.get(replica.replica_id, 0) / 100.0
        cpu_hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, "cpu") or 0.0
        ssd_hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, "ssd") or 0.0
        w = self.layer_weights
        return w["gpu"] * gpu_hit + w["cpu"] * cpu_hit + w["ssd"] * ssd_hit


StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
