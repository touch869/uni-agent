"""Tests for llm_router collectors seam (``RouteDataProvider`` facade).

This is the minimal collaborator seam the Balancer constructs against
(detailed_balancer.md §2.4): construction, lifecycle (start/stop), and the
register/unregister calls the Balancer makes on add/remove. Real collection,
the store, and the per-request KV-cache query are deferred to the collectors-
module detailed design — only the lifecycle/registration seam is real here.
"""

from __future__ import annotations

from uni_agent.llm_router import KVCAwareConfig, KVCAwareStrategyConfig
from uni_agent.llm_router.collectors import ReplicaMetrics, RouteDataProvider


def _config() -> KVCAwareConfig:
    return KVCAwareConfig(
        strategies=[KVCAwareStrategyConfig(weight=1.0, collector_names=["vllm_zmq"])]
    )


# ============================================================
# RouteDataProvider construction + lifecycle
# ============================================================


class TestRouteDataProviderLifecycle:
    """P01-Pnn: construction and lifecycle seam."""

    def test_p01_construction_accepts_config(self):
        """
        Feature: RouteDataProvider constructs from a KVCAwareConfig
        Description: RouteDataProvider(KVCAwareConfig(...))
        Expectation: returns a RouteDataProvider instance
        """
        provider = RouteDataProvider(_config())
        assert isinstance(provider, RouteDataProvider)

    def test_p02_start_stop_callable(self):
        """
        Feature: start/stop lifecycle methods are callable
        Description: call start() then stop() on a provider
        Expectation: both return None without raising
        """
        provider = RouteDataProvider(_config())
        assert provider.start() is None
        assert provider.stop() is None

    def test_p03_get_metrics_returns_default_snapshot(self):
        """
        Feature: get_metrics returns a default (empty) snapshot for any replica
        Description: get_metrics on an unregistered replica id
        Expectation: returns a ReplicaMetrics equal to the empty default
        """
        provider = RouteDataProvider(_config())
        assert provider.get_metrics("s0") == ReplicaMetrics()
