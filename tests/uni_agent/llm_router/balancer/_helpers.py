"""Helpers for balancer unit tests.

Defines ``_FakeProvider`` and helper functions.  Patching is done by
``conftest.py`` via a session-scoped autouse fixture so it never leaks
to Ray workers in other test directories.
"""

from __future__ import annotations

from omegaconf import OmegaConf


class _FakeProvider:
    """Stand-in for RouteDataProvider — no real collectors run."""

    def __init__(self, collectors_config, collection_names, server_addresses=None, kv_event_endpoints=None):
        self.collectors_config = collectors_config
        self.collection_names = collection_names
        self.server_addresses = server_addresses
        self.kv_event_endpoints = kv_event_endpoints
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_metric(self, replica_id, key):
        return 0.0

    def get_metrics(self, replica_id):
        return {}

    def get_gpu_prefix_hit_rate(self, prompt_ids):
        return {}

    def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
        return 0.0


def _router_config(weight: float = 1.0):
    """Build a minimal router_config (OmegaConf) the Balancer accepts."""
    return OmegaConf.create(
        {
            "router_class": "uni_agent.llm_router.balancer.KVCAwareBalancer",
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": weight,
                    "collector_names": ["vllm_zmq"],
                },
            ],
        }
    )


def _fake_init_provider(self):
    """Replacement for KVCAwareBalancer._init_provider in unit tests."""
    collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
    self._provider = _FakeProvider(
        self._config.collector,
        collection_names,
    )
    self._provider.start()


def _fake_resolve_max_num_seqs(servers):
    """Return a fixed capacity — unit tests have string handles, not Ray actors."""
    return 16


def _make_balancer(servers=None):
    """Build a balancer over the given servers (default two)."""
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    if servers is None:
        servers = {"s0": "h0", "s1": "h1"}
    return KVCAwareBalancer(servers, _router_config())
