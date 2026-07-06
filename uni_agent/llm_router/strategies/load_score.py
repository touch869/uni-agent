"""Load-score function for KVCacheAwareStrategy.

Returns ``load ∈ [0, 1]`` (bigger = more loaded).

``max_num_seqs`` is injected by the Balancer via ``set_capacity()``
(fetched from the server handle's rollout config = ``--max-num-seqs``).
"""

from __future__ import annotations

DEFAULT_LOAD_WEIGHTS: tuple[float, float, float] = (0.4, 0.3, 0.3)


def load_normalized(
    kv_usage: float,
    running: int | float,
    waiting: int | float,
    *,
    max_num_seqs: int,
    weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
) -> float:
    """load = a·kv_usage + b·running/max_num_seqs + c·waiting/max_num_seqs"""
    a, b, c = weights
    running_usage = min(1.0, float(running) / float(max_num_seqs))
    waiting_usage = min(1.0, float(waiting) / float(max_num_seqs))
    return a * float(kv_usage) + b * running_usage + c * waiting_usage
