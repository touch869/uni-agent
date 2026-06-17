"""Tests for llm_router strategy seams (runtime strategy classes + registry).

These are the minimal collaborator seams the Balancer constructs against
(see detailed_balancer.md §2.3). Only construction and registry dispatch are
real here; the actual scoring/ranking algorithm lives in a future strategy-
module detailed design, so ``route()`` ranking is a deferred stub.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router import KVCAwareStrategyConfig
from uni_agent.llm_router.strategies import (
    KVCacheAwareStrategy,
    ReplicaInfo,
    StrategyRegistry,
)


# ============================================================
# StrategyRegistry
# ============================================================


class TestStrategyRegistry:
    """R01-Rnn: StrategyRegistry dispatch by config dataclass type."""

    def test_r01_get_returns_registered_strategy_class(self):
        """
        Feature: StrategyRegistry dispatches config type to strategy class
        Description: look up the strategy class registered for KVCAwareStrategyConfig
        Expectation: returns the KVCacheAwareStrategy class object
        """
        cls = StrategyRegistry.get(KVCAwareStrategyConfig)
        assert cls is KVCacheAwareStrategy


# ============================================================
# KVCacheAwareStrategy.from_config
# ============================================================


class TestKVCacheAwareStrategy:
    """R02-Rnn: KVCacheAwareStrategy construction seam."""

    def test_r02_from_config_returns_instance_carrying_config(self):
        """
        Feature: from_config constructs a strategy instance from its config
        Description: KVCacheAwareStrategy.from_config(KVCAwareStrategyConfig(weight=0.7))
        Expectation: returns a KVCacheAwareStrategy carrying the config's strategy fields
        """
        cfg = KVCAwareStrategyConfig(
            weight=0.7, alpha=0.7, load_threshold=80, collector_names=["vllm_zmq"]
        )
        strategy = KVCacheAwareStrategy.from_config(cfg)
        assert isinstance(strategy, KVCacheAwareStrategy)
        assert strategy.alpha == 0.7
        assert strategy.load_threshold == 80


# ============================================================
# ReplicaInfo
# ============================================================


class TestReplicaInfo:
    """R03: ReplicaInfo value type."""

    def test_r03_carries_only_replica_id(self):
        """
        Feature: ReplicaInfo carries a replica_id and no actor handle
        Description: construct ReplicaInfo(replica_id="s0")
        Expectation: ri.replica_id == "s0" and has no handle attribute
        """
        ri = ReplicaInfo(replica_id="s0")
        assert ri.replica_id == "s0"
        assert not hasattr(ri, "handle")


# ============================================================
# route() — deferred ranking entry
# ============================================================


class TestRoute:
    """R04: route() ranking entry (deferred stub)."""

    def test_r04_route_placeholder_returns_replica_ids(self):
        """
        Feature: route() is a placeholder ranking (flow-only; real algorithm deferred)
        Description: call route() with two ReplicaInfo candidates
        Expectation: returns both replica ids (a valid ranking; order not asserted)
        """
        from uni_agent.llm_router.strategies import ReplicaInfo, route

        ranking = route([], [1, 2, 3], None, [ReplicaInfo("s0"), ReplicaInfo("s1")])
        assert set(ranking) == {"s0", "s1"}
