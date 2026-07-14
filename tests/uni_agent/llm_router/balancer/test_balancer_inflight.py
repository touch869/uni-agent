"""Inflight count end-to-end with the REAL inflight_stat collector.

Parallel to test_balancer_sticky.py: ``collector_names`` lists ``inflight_stat``,
so the patched ``_FakeCollectorManager`` builds a real
``Collector(CallbackTransport(self), InflightDecoder)`` that registers
``on_acquire`` / ``on_release`` on the Balancer. acquire bumps the chosen
replica's INFLIGHT_COUNT by +1, release decrements it by -1 — mirroring verl
``GlobalRequestLoadBalancer._inflight_requests``.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestInflightEndToEnd:
    """Real inflight_stat collector: acquire ±1, release ∓1, per-replica isolation."""

    def test_inflight_defaults_to_zero(self):
        balancer = _make_balancer({"s0": "h0"})
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0

    def test_acquire_increments_inflight(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        assert balancer._store.get_metric(sid, MetricKey.INFLIGHT_COUNT) == 1

    def test_release_decrements_inflight(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        balancer.release_server(sid)
        assert balancer._store.get_metric(sid, MetricKey.INFLIGHT_COUNT) == 0

    def test_acquire_release_symmetric_returns_to_zero(self):
        balancer = _make_balancer({"s0": "h0"})
        balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 2
        balancer.release_server("s0")
        balancer.release_server("s0")
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0

    def test_inflight_isolated_per_replica(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.acquire_server("r1", [1])  # cold-start tie-break → s0
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
        assert balancer._store.get_metric("s1", MetricKey.INFLIGHT_COUNT) == 0
