"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCacheAwareStrategy``) live under ``uni_agent.llm_router.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from uni_agent.llm_router.config.base import ConfigError, StrategyConfig, _multiline_repr
from uni_agent.llm_router.types import Layer


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load
    """

    alpha: float = 0.7
    load_threshold: float = 0.9
    layer_weights: dict[Layer, float] = field(
        default_factory=lambda: {Layer.GPU: 0.7, Layer.CPU: 0.2, Layer.SSD: 0.1}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")
        # Normalize yaml str keys → Layer (also validates each key is a known layer).
        try:
            self.layer_weights = {Layer(k): v for k, v in self.layer_weights.items()}
        except ValueError:
            raise ConfigError(f"layer_weights keys must be layer names, got {set(self.layer_weights)}")
        if set(self.layer_weights.keys()) != {Layer.GPU, Layer.CPU, Layer.SSD}:
            raise ConfigError(
                f"layer_weights must be exactly {{{Layer.GPU}, {Layer.CPU}, {Layer.SSD}}}, "
                f"got {set(self.layer_weights.keys())}"
            )
        weights_sum = sum(self.layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise ConfigError(f"layer_weights values must sum to 1.0, got {weights_sum}")

    __repr__ = _multiline_repr
