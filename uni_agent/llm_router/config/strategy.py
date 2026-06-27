"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCAwareStrategy``) live under ``uni_agent.llm_router.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from uni_agent.llm_router.config.base import ConfigError, StrategyConfig, _multiline_repr


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load

    When ``use_capacity_load=False`` (legacy mode):

      S_load = (1 - kv_cache_usage_perc) / (1 + running + waiting)

      Overload is determined by ``norm(S_load) < load_threshold`` (relative).

    When ``use_capacity_load=True`` (capacity mode):

      where running_ratio = running / max_num_seqs (from env MAX_NUM_SEQS).

      S_load = w_kv × (1 - kv_usage) + w_run × (1 - running_ratio)
             + w_queue × (1 - queue_fraction)
      where queue_fraction = waiting / (running + waiting + 1).

      S_load ∈ [0, 1] with absolute physical meaning ("remaining effective capacity").
    """

    alpha: float = 0.7
    load_threshold: float = 0.1

    # -- Capacity mode toggle and parameters --
    use_capacity_load: bool = False
    w_kv: float = 0.5
    w_run: float = 0.3
    w_queue: float = 0.2

    layer_weights: dict[str, float] = field(default_factory=lambda: {"cpu": 1.0, "ssd": 0.25})

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")
        valid_keys = {"cpu", "ssd"}
        if not set(self.layer_weights.keys()) == valid_keys:
            raise ConfigError(
                f"layer_weights keys must be {valid_keys} only, got {set(self.layer_weights.keys())}"
            )
        if self.use_capacity_load:
            if self.w_kv < 0 or self.w_run < 0 or self.w_queue < 0:
                raise ConfigError(
                    f"w_kv, w_run, w_queue must be >= 0, got "
                    f"w_kv={self.w_kv}, w_run={self.w_run}, w_queue={self.w_queue}"
                )
            if not (0.99 <= self.w_kv + self.w_run + self.w_queue <= 1.01):
                raise ConfigError(
                    f"w_kv + w_run + w_queue must sum to ~1.0, got "
                    f"{self.w_kv + self.w_run + self.w_queue}"
                )

    __repr__ = _multiline_repr
