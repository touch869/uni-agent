"""KVCache-aware runtime strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from uni_agent.llm_router.collectors.metric_spec import MetricKey
from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from uni_agent.llm_router.collectors.provider import RouteDataProvider
    from uni_agent.llm_router.strategies.base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

# Environment variable for vLLM max_num_seqs (batch concurrency capacity).
_DEFAULT_MAX_NUM_SEQS = 1024


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(self, cfg: KVCAwareStrategyConfig) -> None:
        # Config-level validation already happened in __post_init__,
        # but we re-validate critical bounds here for defense-in-depth.
        self._cfg_check(cfg)

        self.alpha = float(cfg.alpha)
        self.load_threshold = float(cfg.load_threshold)
        self.use_capacity_load = cfg.use_capacity_load
        self.w_kv = float(cfg.w_kv)
        self.w_run = float(cfg.w_run)
        self.w_queue = float(cfg.w_queue)
        self.layer_weights = dict(cfg.layer_weights)
        self.collector_names = list(cfg.collector_names)
        self.weight = float(cfg.weight)

        # Read max_num_seqs from environment variable at init time.
        self.max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", _DEFAULT_MAX_NUM_SEQS))
        if self.max_num_seqs <= 0:
            raise StrategyError(f"MAX_NUM_SEQS must be > 0, got {self.max_num_seqs}")

        # Tiers for which we've already warned about missing slow-path data,
        # so the slow-path degradation notice is emitted once, not per request.
        self._tier_warned: set[str] = set()

        mode_tag = "capacity" if self.use_capacity_load else "legacy"
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, mode={mode_tag}, "
            f"max_num_seqs={self.max_num_seqs}, layer_weights={self.layer_weights}",
        )
        if self.use_capacity_load:
            logger.info(
                f"w_kv={self.w_kv:.2f}, w_run={self.w_run:.2f}, w_queue={self.w_queue:.2f}",
            )

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> "KVCacheAwareStrategy":
        """Construct a strategy instance from a parsed config."""
        return cls(cfg)

    def _cfg_check(self, cfg):
        """check the configuration"""
        if not 0 <= cfg.alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {cfg.alpha}")
        if not 0 < cfg.load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {cfg.load_threshold}")
        _valid_tiers = {"cpu", "ssd"}
        for tier, tier_weight in cfg.layer_weights.items():
            if tier not in _valid_tiers:
                raise StrategyError(f"layer_weights key must be in {_valid_tiers}, got '{tier}'")
            if tier_weight < 0:
                raise StrategyError(f"layer_weights[{tier}] must be >= 0, got {tier_weight}")

    # -- Capacity-based load helpers ------------------------------------------

    def _capacity_load_score(
        self,
        kv_usage: float,
        running: int,
        waiting: int,
    ) -> float:
        """Compute S_load using the capacity-based weighted formula.

        S_load = w_kv × (1 - kv_usage) + w_run × (1 - running_ratio)
               + w_queue × (1 - queue_fraction)

        where:
          running_ratio  = running / max_num_seqs              ∈ [0, 1]
          queue_fraction = waiting / (running + waiting + 1)   ∈ [0, 1)

        Output ∈ [0, 1] with absolute meaning: "remaining effective capacity".
        """
        running_ratio = min(float(running) / self.max_num_seqs, 1.0)
        queue_fraction = float(waiting) / (float(running) + float(waiting) + 1.0)
        return (
            self.w_kv * (1.0 - kv_usage)
            + self.w_run * (1.0 - running_ratio)
            + self.w_queue * (1.0 - queue_fraction)
        )

    # -- Main scoring ---------------------------------------------------------

    def score(
        self,
        prompt_ids: list[int] | None,
        provider: "RouteDataProvider",
        replicas: list["ReplicaInfo"],
    ) -> list[float]:
        """Score each replica. Larger is better.

        Legacy mode (use_capacity_load=False):

          S_load = (1 - kv_cache_usage_perc) / (1 + running + waiting)
          Overloaded when norm(S_load) < load_threshold (relative).
          S = alpha * norm(S_cache) + (1 - alpha) * norm(S_load)

        Capacity mode (use_capacity_load=True):

          S_load = w_kv × (1-kv) + w_run × (1-running_ratio)
                 + w_queue × (1-queue_fraction)  ∈ [0, 1]
          No normalization — raw scores combined directly:
          S = alpha * S_cache + (1 - alpha) * S_load
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []

        effective_prompt_ids = prompt_ids or []

        # ── Phase 1: Compute raw load scores ──────────────────────────────
        load_scores: list[float] = []
        is_overloaded: list[bool] = []

        for replica in replicas:
            replica_metrics = provider.get_metrics(replica.replica_id)
            kv_usage = float(replica_metrics.get(MetricKey.KV_CACHE_USAGE_PERC, 0.0))
            running = int(replica_metrics.get(MetricKey.NUM_REQUESTS_RUNNING, 0))
            waiting = int(replica_metrics.get(MetricKey.NUM_REQUESTS_WAITING, 0))

            if self.use_capacity_load:
                s_load = self._capacity_load_score(kv_usage, running, waiting)
            else:
                s_load = (1.0 - kv_usage) / (1.0 + float(running) + float(waiting))
            
            overloaded = (1 - s_load) > self.load_threshold

            load_scores.append(s_load)
            is_overloaded.append(overloaded)
            logger.debug(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} "
                f"running={running} waiting={waiting} raw_s_load={s_load:.4f} "
                f"overloaded={overloaded}",
            )

        # ── Phase 2: Determine overload status ────────────────────────────
        all_overloaded = all(is_overloaded)
        overloaded_count = sum(is_overloaded)

        if overloaded_count > 0 and not all_overloaded:
            logger.info(
                f"score(): {overloaded_count}/{len(replicas)} replicas overloaded, "
                f"routing to healthy subset",
            )
        elif all_overloaded:
            logger.info(
                f"score(): all {len(replicas)} replicas overloaded, "
                f"scoring with full set (fallback)",
            )

        logger.debug(f"score(): S_load={[f'{v:.4f}' for v in load_scores]}")
        logger.debug(f"score(): overloaded={is_overloaded}")

        # ── Phase 3: Compute raw cache scores ─────────────────────────────
        cache_scores: list[float] = []

        if all_overloaded:
            # All overloaded → slow path for every replica.
            for idx, replica in enumerate(replicas):
                cache_scores.append(self._slow_cache(provider, replica, effective_prompt_ids))
        else:
            # Some healthy replicas: query GPU hit for all at once.
            # get_gpu_prefix_hit_rate returns {replica_id: 0-100}; scale to 0-1.
            gpu_hit_pct = provider.get_gpu_prefix_hit_rate(effective_prompt_ids)
            gpu_hits = [
                0.0 if is_overloaded[idx]
                else gpu_hit_pct.get(replica.replica_id, 0) / 100.0
                for idx, replica in enumerate(replicas)
            ]
            use_fast = any(
                gpu_hits[idx] > 0 for idx in range(len(replicas)) if not is_overloaded[idx]
            )
            logger.debug(
                f"score(): path={'fast (GPU hit)' if use_fast else 'slow (tier cache)'}"
            )

            for idx, replica in enumerate(replicas):
                if is_overloaded[idx]:
                    cache_scores.append(0.0)
                else:
                    s_cache = (
                        gpu_hits[idx]
                        if use_fast
                        else self._slow_cache(provider, replica, effective_prompt_ids)
                    )
                    cache_scores.append(s_cache)

        logger.debug(f"score(): S_cache={[f'{v:.4f}' for v in cache_scores]}")

        # ── Phase 4: Combine scores ───────────────────────────────────────
        if self.use_capacity_load:
            # Capacity mode: raw scores combined directly (no normalization).
            # S_load ∈ [0, 1] has absolute meaning; S_cache ∈ [0, ~1] for fast
            # path, can exceed 1 for slow path — alpha controls the trade-off.
            result = [
                self.alpha * cache_scores[idx] + (1 - self.alpha) * load_scores[idx]
                for idx in range(len(replicas))
            ]
        else:
            result = [
                self.alpha * cache_scores[idx] + (1 - self.alpha) * load_scores[idx]
                for idx in range(len(replicas))
            ]

        logger.info(
            f"score(): final scores "
            + ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas)),
        )
        return result

    def _slow_cache(
        self,
        provider: "RouteDataProvider",
        replica: "ReplicaInfo",
        prompt_ids: list[int],
    ) -> float:
        """Weighted tier hit rate from GPU eviction table (slow path).

        cpu and ssd tiers are mutually exclusive, so the weighted sum is <= 1
        with default weights {"cpu": 1.0, "ssd": 0.25}.

        If a tier's hit rate is unavailable (``None`` — mooncake tier collector
        not yet implemented), its contribution degrades to 0 and a one-time
        WARNING is emitted per tier.
        """
        total = 0.0
        for tier, tier_weight in self.layer_weights.items():
            hit = provider.get_tier_prefix_hit_rate(replica.replica_id, prompt_ids, tier)
            if hit is None:
                if tier not in self._tier_warned:
                    logger.warning(
                        f"slow path: tier='{tier}' prefix hit unavailable "
                        f"(mooncake tier collector not implemented) — "
                        f"degrading S_cache['{tier}'] to 0",
                    )
                    self._tier_warned.add(tier)
                continue
            total += hit * tier_weight
        return total


# Auto-register: config dataclass type → runtime strategy class.
StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
