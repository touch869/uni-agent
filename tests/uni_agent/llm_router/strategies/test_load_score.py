"""Unit tests for the load-score function (strategies/load_score.py).

Returns ``load ∈ [0,1]`` with the convention **bigger = more loaded**
(the strategy converts via ``s_load = 1 - load``).

Normalized formula:
    running_usage = min(1, running / max_num_seqs)
    waiting_usage = min(1, waiting / max_num_seqs)
    load = a·kv + b·running_usage + c·waiting_usage      (a+b+c=1)
Default weights (a, b, c) = (0.4, 0.3, 0.3).

``max_num_seqs`` is injected by the Balancer via ``set_capacity()``
(fetched from the server handle's rollout config).
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.strategies.load_score import (
    DEFAULT_LOAD_WEIGHTS,
    load_normalized,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestLoadNormalized:
    def test_idle_replica_is_zero(self):
        assert load_normalized(0.0, 0, 0, max_num_seqs=64) == pytest.approx(0.0)

    def test_kv_only_contribution(self):
        assert load_normalized(0.5, 0, 0, max_num_seqs=64) == pytest.approx(0.2)

    def test_running_and_kv(self):
        assert load_normalized(0.5, 32, 0, max_num_seqs=64) == pytest.approx(0.35)

    def test_running_clamped_to_one(self):
        assert load_normalized(0.8, 128, 0, max_num_seqs=64) == pytest.approx(0.62)

    def test_waiting_term(self):
        assert load_normalized(0.0, 0, 10, max_num_seqs=64) == pytest.approx(0.3 * (10 / 64))

    def test_custom_weights_change_load(self):
        load = load_normalized(0.5, 32, 0, max_num_seqs=64, weights=(0.6, 0.2, 0.2))
        assert load == pytest.approx(0.4)
        assert load != pytest.approx(0.35)

    def test_near_saturated_exceeds_threshold(self):
        load = load_normalized(1.0, 64, 1000, max_num_seqs=64)
        assert load > 0.9
        assert load <= 1.0


class TestDefaultWeights:
    def test_default_weights_tuple(self):
        assert DEFAULT_LOAD_WEIGHTS == (0.4, 0.3, 0.3)
        assert sum(DEFAULT_LOAD_WEIGHTS) == pytest.approx(1.0)
