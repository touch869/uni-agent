"""Unit tests for the LLM router strategy module.

Covers the test matrix in
``router_doc/design/detailed_strategy_module.md`` §8. Numeric baselines reuse
the spec's partial/full-overload scenario tables (threshold=80, alpha=0.7,
1-alpha=0.3, layer_weights={cpu:1.0, ssd:0.25}).

A ``FakeMetricsProvider`` stands in for the future real provider; loads are made
readable by fixing ``gpu_util=1.0`` so ``load == queue_depth``.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.strategy import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)
from uni_agent.llm_router.types import ReplicaInfo, RouteContext


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeMetricsProvider:
    """In-memory metrics. data[replica_id] = {queue, gpu_util, gpu_hit, tiers}."""

    def __init__(self, data: dict[str, dict]):
        self._data = data

    def get_gpu_utilization(self, replica_id: str) -> float:
        return self._data.get(replica_id, {}).get("gpu_util", 0.0)

    def get_queue_depth(self, replica_id: str) -> int:
        return self._data.get(replica_id, {}).get("queue", 0)

    def get_gpu_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int]) -> float:
        return self._data.get(replica_id, {}).get("gpu_hit", 0.0)

    def get_tier_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], tier: str) -> float:
        return self._data.get(replica_id, {}).get("tiers", {}).get(tier, 0.0)


class ConstantStrategy:
    """Returns a fixed per-replica score list (for route() composition tests)."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, ctx, metrics, replicas) -> list[float]:
        return list(self._scores)


class BadLengthStrategy:
    """Returns a wrong-length list to exercise the contract check in route()."""

    def score(self, ctx, metrics, replicas) -> list[float]:
        return [1.0]


class RaisingStrategy:
    """Raises inside score() to exercise route()'s exception wrapping."""

    def score(self, ctx, metrics, replicas) -> list[float]:
        raise KeyError("boom")


CTX = RouteContext(request_id="req-1", prompt_ids=[1, 2, 3])


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=i) for i in ids]


def _load(queue: int, **extra) -> dict:
    """Build a replica metrics dict with gpu_util=1.0 so load == queue."""
    return {"gpu_util": 1.0, "queue": queue, **extra}


# --------------------------------------------------------------------------- #
# StrategyRegistry
# --------------------------------------------------------------------------- #
class TestStrategyRegistry:
    def test_builtin_registered(self):
        assert StrategyRegistry.get("kv_cache_aware") is KVCacheAwareStrategy

    def test_register_and_get(self):
        name = "test_dummy_strategy_xyz"

        @StrategyRegistry.register(name)
        class _Dummy:
            def score(self, ctx, metrics, replicas):
                return [0.0] * len(replicas)

        try:
            assert StrategyRegistry.get(name) is _Dummy
        finally:
            StrategyRegistry._registry.pop(name, None)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):

            @StrategyRegistry.register("kv_cache_aware")
            class _Dup:
                pass

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            StrategyRegistry.get("no_such_strategy")

    def test_list_strategies_sorted(self):
        names = StrategyRegistry.list_strategies()
        assert names == sorted(names)
        assert "kv_cache_aware" in names


# --------------------------------------------------------------------------- #
# KVCacheAwareStrategy — load & overload
# --------------------------------------------------------------------------- #
class TestKVCacheAwareLoad:
    def test_s_load_boundaries(self):
        # load=0 -> S_load=1 ; load=threshold -> 0 ; load>threshold -> <0
        strat = KVCacheAwareStrategy()  # alpha=0.7, threshold=80
        m = FakeMetricsProvider(
            {
                "empty": _load(0),  # S_load=1
                "crit": _load(80),  # S_load=0 (not overloaded)
                "over": _load(120),  # S_load=-0.5 (overloaded)
            }
        )
        scores = strat.score(CTX, m, _replicas("empty", "crit", "over"))
        # all have gpu_hit=0 -> slow path, cache=0
        # empty (not overloaded): 0.7*0 + 0.3*1 = 0.3
        # crit  (not overloaded): 0.7*0 + 0.3*0 = 0.0
        # over  (overloaded):     S_load = -0.5  (partial-overload uses S_load directly)
        assert scores == pytest.approx([0.3, 0.0, -0.5])

    def test_threshold_is_not_overload(self):
        # load == threshold is kept (S_load=0), not filtered as overloaded.
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"crit": _load(80, gpu_hit=0.5), "over": _load(120, gpu_hit=0.9)})
        scores = strat.score(CTX, m, _replicas("crit", "over"))
        # crit not overloaded, gpu_hit>0 in non-overloaded -> fast path
        # crit: 0.7*0.5 + 0.3*0 = 0.35 ; over: S_load = 1-120/80 = -0.5
        assert scores == pytest.approx([0.35, -0.5])
        assert scores[0] > 0 > scores[1]

    def test_missing_metrics_optimistic(self):
        # Unknown replica -> load 0 -> S_load 1, not overloaded.
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({})  # nothing configured
        scores = strat.score(CTX, m, _replicas("ghost"))
        assert scores == pytest.approx([0.3])  # 0.7*0 + 0.3*1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 1.5},
            {"alpha": -0.1},
            {"load_threshold": 0},
            {"load_threshold": -5},
            {"layer_weights": {"cpu": -1.0}},             # negative weight
            {"layer_weights": {"nvme": 1.0}},             # invalid tier key
            {"layer_weights": {"cpu": 1.0, "nvme": 0.5}},  # mixed valid+invalid
            {"weight": 0},
            {"weight": -0.2},
            {"weight": 1.5},  # upper bound (spec §6.7: weight in (0, 1])
        ],
    )
    def test_construction_validation(self, kwargs):
        with pytest.raises(StrategyError):
            KVCacheAwareStrategy(**kwargs)

    def test_empty_layer_weights_honored(self):
        # An explicit empty dict disables slow-path tiers; must not be replaced
        # by the default {cpu,ssd}.
        strat = KVCacheAwareStrategy(layer_weights={})
        assert strat.layer_weights == {}
        m = FakeMetricsProvider({"a": _load(0, gpu_hit=0.0, tiers={"cpu": 1.0, "ssd": 1.0})})
        # slow path, but no tiers configured -> cache 0 -> 0.7*0 + 0.3*1 = 0.3
        assert strat.score(CTX, m, _replicas("a")) == pytest.approx([0.3])

    def test_alpha_one_overloaded_still_sinks_below_healthy(self):
        # Regression: with alpha=1.0 the overloaded replica must still rank below
        # a healthy zero-cache replica (both used to tie at 0.0).
        strat = KVCacheAwareStrategy(alpha=1.0)
        m = FakeMetricsProvider(
            {
                "over": _load(120, gpu_hit=1.0, tiers={"cpu": 1.0}),  # overloaded, high cache
                "healthy": _load(40, gpu_hit=0.0, tiers={"cpu": 0.0}),  # healthy, zero cache
            }
        )
        replicas = _replicas("over", "healthy")
        scores = strat.score(CTX, m, replicas)
        # over: S_load = 1 - 120/80 = -0.5 ; healthy: 1.0*0 = 0.0
        assert scores[0] < 0 <= scores[1]
        assert route([(strat, 1.0)], CTX, m, replicas) == [1, 0]  # healthy first


# --------------------------------------------------------------------------- #
# KVCacheAwareStrategy — fast/slow path selection
# --------------------------------------------------------------------------- #
class TestKVCacheAwareFastSlow:
    def test_no_overload_with_gpu_hit_is_fast(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {"a": _load(40, gpu_hit=0.8, tiers={"cpu": 1.0}), "b": _load(20, gpu_hit=0.0, tiers={"cpu": 1.0})}
        )
        scores = strat.score(CTX, m, _replicas("a", "b"))
        # fast path -> cache = gpu_hit (tiers ignored)
        # a: 0.7*0.8 + 0.3*0.5 = 0.71 ; b: 0.7*0.0 + 0.3*0.75 = 0.225
        assert scores == pytest.approx([0.71, 0.225])

    def test_no_overload_no_gpu_hit_is_slow(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {
                "a": _load(40, gpu_hit=0.0, tiers={"cpu": 0.6, "ssd": 0.2}),
                "b": _load(20, gpu_hit=0.0, tiers={"cpu": 0.3, "ssd": 0.4}),
            }
        )
        scores = strat.score(CTX, m, _replicas("a", "b"))
        # slow cache: a=0.6+0.05=0.65 ; b=0.3+0.1=0.4
        # a: 0.7*0.65 + 0.3*0.5 = 0.605 ; b: 0.7*0.4 + 0.3*0.75 = 0.505
        assert scores == pytest.approx([0.605, 0.505])

    def test_partial_overload_only_considers_healthy_subset(self):
        # Overloaded replica has gpu_hit>0, but only the healthy subset decides
        # fast/slow. Healthy replicas have no gpu hit -> slow path.
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {
                "a": _load(40, gpu_hit=0.0, tiers={"cpu": 0.6}),
                "over": _load(120, gpu_hit=0.9, tiers={"cpu": 0.9}),
            }
        )
        scores = strat.score(CTX, m, _replicas("a", "over"))
        # slow path. a (healthy): 0.7*0.6 + 0.3*0.5 = 0.57
        # over (overloaded): S_load = 1-120/80 = -0.5  (cache ignored)
        assert scores == pytest.approx([0.57, -0.5])

    def test_all_overloaded_forces_slow_even_with_gpu_hit(self):
        # All overloaded -> forced slow path: ranking follows tier cache, not gpu_hit.
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {
                "a": _load(120, gpu_hit=0.8, tiers={"cpu": 0.6, "ssd": 0.2}),  # slow 0.65
                "b": _load(100, gpu_hit=0.5, tiers={"cpu": 0.9}),  # slow 0.9
                "c": _load(160, gpu_hit=0.9, tiers={"cpu": 0.3, "ssd": 0.4}),  # slow 0.4
            }
        )
        scores = strat.score(CTX, m, _replicas("a", "b", "c"))
        # all overloaded -> score = alpha * slow_cache (load dropped)
        # a: 0.7*0.65=0.455 ; b: 0.7*0.9=0.63 ; c: 0.7*0.4=0.28
        assert scores == pytest.approx([0.455, 0.63, 0.28])
        # 'c' has the highest gpu_hit (0.9) but lowest score -> proves forced slow.
        ranking = route([(strat, 1.0)], CTX, m, _replicas("a", "b", "c"))
        assert ranking == [1, 0, 2]


# --------------------------------------------------------------------------- #
# KVCacheAwareStrategy — combined scoring
# --------------------------------------------------------------------------- #
class TestKVCacheAwareScore:
    def test_overloaded_sinks_below_healthy_despite_high_cache(self):
        # Overloaded replica with very high cache must still rank below healthy.
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {
                "healthy": _load(40, gpu_hit=0.0, tiers={"cpu": 0.0}),  # 0.7*0 + 0.3*0.5 = 0.15
                "over": _load(120, gpu_hit=1.0, tiers={"cpu": 1.0}),  # overloaded -> S_load = -0.5
            }
        )
        scores = strat.score(CTX, m, _replicas("healthy", "over"))
        assert scores[0] >= 0 > scores[1]
        assert route([(strat, 1.0)], CTX, m, _replicas("healthy", "over")) == [0, 1]

    def test_slow_cache_truncated_to_one(self):
        strat = KVCacheAwareStrategy()
        # cpu=1.0, ssd=1.0 -> 1.0 + 0.25 = 1.25 -> min(1.0)
        m = FakeMetricsProvider({"a": _load(0, gpu_hit=0.0, tiers={"cpu": 1.0, "ssd": 1.0})})
        scores = strat.score(CTX, m, _replicas("a"))
        # truncated cache=1.0 -> 0.7*1.0 + 0.3*1.0 = 1.0 (untruncated would be 1.175)
        assert scores == pytest.approx([1.0])

    def test_ssd_zero_hit_contributes_nothing(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"a": _load(0, gpu_hit=0.0, tiers={"cpu": 0.5, "ssd": 0.0})})
        scores = strat.score(CTX, m, _replicas("a"))
        # slow cache = 0.5*1.0 + 0.0*0.25 = 0.5 -> 0.7*0.5 + 0.3*1 = 0.65
        assert scores == pytest.approx([0.65])


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
class TestStrategyContract:
    def test_protocol_satisfied(self):
        assert isinstance(KVCacheAwareStrategy(), RoutingStrategy)

    def test_score_signature_callable(self):
        # isinstance against a runtime_checkable Protocol only checks method-name
        # presence, not arity; exercise the real 3-arg call to pin the signature.
        import inspect

        params = list(inspect.signature(KVCacheAwareStrategy.score).parameters)
        assert params == ["self", "ctx", "metrics", "replicas"]
        # and it actually runs with the documented call shape
        strat = KVCacheAwareStrategy()
        assert strat.score(CTX, FakeMetricsProvider({}), _replicas("a")) is not None

    def test_output_length_and_order(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"a": _load(20, gpu_hit=0.9), "b": _load(40, gpu_hit=0.1)})
        replicas = _replicas("a", "b")
        scores = strat.score(CTX, m, replicas)
        assert len(scores) == len(replicas)
        assert scores[0] > scores[1]  # a: lower load + higher hit

    def test_stateless_repeatable(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"a": _load(40, gpu_hit=0.8), "b": _load(20, gpu_hit=0.5)})
        replicas = _replicas("a", "b")
        first = strat.score(CTX, m, replicas)
        second = strat.score(CTX, m, replicas)
        assert first == pytest.approx(second)


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #
class TestRoute:
    def test_single_strategy_descending(self):
        m = FakeMetricsProvider({})
        ranking = route([(ConstantStrategy([0.2, 0.5, 0.1]), 1.0)], CTX, m, _replicas("a", "b", "c"))
        assert ranking == [1, 0, 2]

    def test_multi_strategy_weighted_sum(self):
        m = FakeMetricsProvider({})
        strategies = [
            (ConstantStrategy([1.0, 2.0, 3.0]), 0.5),
            (ConstantStrategy([3.0, 1.0, 0.0]), 0.5),
        ]
        # final = [2.0, 1.5, 1.5] -> [0, 1, 2] (stable on tie)
        ranking = route(strategies, CTX, m, _replicas("a", "b", "c"))
        assert ranking == [0, 1, 2]

    def test_overloaded_last_but_present(self):
        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider(
            {
                "a": _load(40, gpu_hit=0.8),
                "b": _load(20, gpu_hit=0.5),
                "over": _load(120, gpu_hit=0.9),
            }
        )
        ranking = route([(strat, 1.0)], CTX, m, _replicas("a", "b", "over"))
        assert ranking == [0, 1, 2]
        assert set(ranking) == {0, 1, 2}  # overloaded still in ranking

    def test_empty_pool_raises(self):
        with pytest.raises(RuntimeError):
            route([(ConstantStrategy([]), 1.0)], CTX, FakeMetricsProvider({}), [])

    def test_length_mismatch_raises(self):
        with pytest.raises(StrategyError):
            route([(BadLengthStrategy(), 1.0)], CTX, FakeMetricsProvider({}), _replicas("a", "b"))

    def test_strategy_exception_wrapped_with_name(self):
        # route() wraps a strategy exception as StrategyError carrying the class
        # name, so multi-strategy failures are attributable.
        with pytest.raises(StrategyError) as exc_info:
            route([(RaisingStrategy(), 1.0)], CTX, FakeMetricsProvider({}), _replicas("a"))
        assert "RaisingStrategy" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_nan_score_ranked_last(self):
        # A NaN score must not land at the top of the ranking.
        m = FakeMetricsProvider({})
        ranking = route(
            [(ConstantStrategy([float("nan"), 0.1, 0.5]), 1.0)],
            CTX,
            m,
            _replicas("nan", "low", "high"),
        )
        assert ranking[0] == 2  # high
        assert ranking[-1] == 0  # nan ranked last


# --------------------------------------------------------------------------- #
# Warnings (spec §7)
# --------------------------------------------------------------------------- #
class TestWarnings:
    def test_all_overloaded_no_cache_warns(self, caplog):
        import logging

        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"a": _load(120), "b": _load(160)})  # all overloaded, no cache
        with caplog.at_level(logging.WARNING, logger="uni_agent.llm_router.strategy"):
            strat.score(CTX, m, _replicas("a", "b"))
        assert any("undifferentiated" in r.message or "overloaded" in r.message for r in caplog.records)

    def test_all_overloaded_with_cache_no_warn(self, caplog):
        import logging

        strat = KVCacheAwareStrategy()
        m = FakeMetricsProvider({"a": _load(120, tiers={"cpu": 0.5}), "b": _load(160)})
        with caplog.at_level(logging.WARNING, logger="uni_agent.llm_router.strategy"):
            strat.score(CTX, m, _replicas("a", "b"))
        assert not caplog.records  # cache signal present -> no warning


# --------------------------------------------------------------------------- #
# Registry idempotency (re-import safety)
# --------------------------------------------------------------------------- #
class TestRegistryIdempotency:
    def test_reregister_same_class_is_idempotent(self):
        existing = StrategyRegistry.get("kv_cache_aware")
        # Re-registering the identical class must not raise (re-import safety).
        StrategyRegistry.register("kv_cache_aware")(existing)
        assert StrategyRegistry.get("kv_cache_aware") is existing

    def test_reregister_different_class_raises(self):
        with pytest.raises(ValueError):

            @StrategyRegistry.register("kv_cache_aware")
            class _Other:
                def score(self, ctx, metrics, replicas):
                    return []


# --------------------------------------------------------------------------- #
# from_config() classmethod
# --------------------------------------------------------------------------- #
class TestFromConfig:
    def test_from_config_correct_fields(self):
        from uni_agent.llm_router.strategies.kvc_aware import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(alpha=0.6, load_threshold=60.0, layer_weights={"cpu": 0.8, "ssd": 0.1}, weight=0.9)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.alpha == pytest.approx(0.6)
        assert strat.load_threshold == pytest.approx(60.0)
        assert strat.layer_weights == {"cpu": 0.8, "ssd": 0.1}
        assert strat.weight == pytest.approx(0.9)

    def test_from_config_scores_match_direct(self):
        from uni_agent.llm_router.strategies.kvc_aware import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(alpha=0.7, load_threshold=80.0, layer_weights={"cpu": 1.0, "ssd": 0.25}, weight=1.0)
        strat_cfg = KVCacheAwareStrategy.from_config(cfg)
        strat_direct = KVCacheAwareStrategy(alpha=0.7, load_threshold=80.0, layer_weights={"cpu": 1.0, "ssd": 0.25}, weight=1.0)
        m = FakeMetricsProvider({"a": _load(40, gpu_hit=0.8), "b": _load(120)})
        replicas = _replicas("a", "b")
        assert strat_cfg.score(CTX, m, replicas) == pytest.approx(strat_direct.score(CTX, m, replicas))
