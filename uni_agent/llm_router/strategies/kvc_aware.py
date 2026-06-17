"""KVCache-aware runtime strategy.

In the full implementation this scores replicas as
``S = α·S_cache + (1-α)·S_load``. The scoring algorithm and the per-request
KV-cache prefix-hit query are deferred to the strategy-module detailed design;
this module provides only the construction seam (``from_config``) the Balancer
wires up (see detailed_balancer.md §2.3).
"""

from __future__ import annotations

from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.llm_router.strategies.registry import StrategyRegistry


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
    ) -> None:
        self.alpha = alpha
        self.load_threshold = load_threshold
        self.layer_weights = layer_weights
        self.collector_names = collector_names
        self.weight = weight

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> "KVCacheAwareStrategy":
        """Construct a strategy instance carrying its parsed config fields."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
        )


# Auto-register: config dataclass type → runtime strategy class.
StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
