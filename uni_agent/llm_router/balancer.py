"""KVCAwareBalancer — top-level orchestration shell for the KVCAware router.

A **pure framework shell** (detailed_balancer.md §1): it wires Config /
Strategy / collectors, manages their lifecycle, and delegates each request to
``route()``. It contains no routing algorithm.

VeRL imports this class by FQN (``router_class``) and wraps it with
``ray.remote(...)`` at runtime, so this is a plain class — directly
constructible and unit-testable. It satisfies the ``RequestLoadBalancer``
Protocol (6 methods) via structural subtyping.
"""

from __future__ import annotations

from typing import Any

import ray

from uni_agent.llm_router.collectors.manager import CollectorManager
from uni_agent.llm_router.config import KVCAwareConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.store import DataStore
from uni_agent.llm_router.strategies import (
    ReplicaInfo,
    StickySessionTable,
    StrategyRegistry,
    route,
)

logger = get_router_logger("balancer")


class KVCAwareBalancer:
    """Pure-framework router shell. See module docstring."""

    def __init__(self, servers: dict[str, Any], router_config: Any) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        self._config = KVCAwareConfig.from_config(router_config)
        self._strategies: list[tuple[Any, float]] = [
            (StrategyRegistry.get(type(cfg)).from_config(cfg), cfg.weight) for cfg in self._config.strategies
        ]
        max_num_seqs = self._resolve_max_num_seqs(servers)
        for strategy, _ in self._strategies:
            if hasattr(strategy, "set_capacity"):
                strategy.set_capacity(max_num_seqs)
        logger.info(f"KVCAwareBalancer: injected max_num_seqs={max_num_seqs} from server handle")
        self._servers: dict[str, Any] = dict(servers)
        self._route_calls = 0
        self._sticky = StickySessionTable(max_size=self._config.sticky_max_size)
        # _store before _init_manager: real env's _init_manager only wires the
        # collector manager, but tests inject a fake store via _init_manager, so
        # the real DataStore must be constructed first (and thus overridable).
        self._store = DataStore()
        self._init_manager()

    @staticmethod
    def _resolve_max_num_seqs(servers: dict[str, Any]) -> int:
        """Fetch ``max_num_seqs`` from a server handle's rollout config."""
        handle = next(iter(servers.values()))
        cfg = ray.get(handle.get_rollout_config.remote())
        value = int(getattr(cfg, "max_num_seqs", 0))
        if value <= 0:
            raise ValueError(f"server handle returned invalid max_num_seqs={value}")
        return value

    def _init_manager(self) -> None:
        """Resolve per-server endpoints from Ray actor handles and init the manager.

        Iterates ``self._servers``, calling ``get_server_address.remote()`` and
        ``get_kv_events_endpoints.remote()`` on each handle to dynamically
        discover the Prometheus polling addresses and ZMQ kv-event endpoints.
        The resolved addresses are then passed to ``CollectorManager``, which
        routes them to the appropriate collector type at creation time.

        Handles that are not real Ray actors (e.g. plain strings passed by
        unit tests or bring-up stubs) have no ``get_server_address`` remote;
        for those, dynamic discovery is skipped and collectors fall back to
        their configured/default endpoints.
        """
        collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
        server_addresses: dict[str, str] = {}
        kv_event_endpoints: dict[str, list[str]] = {}
        addr_futures = []
        ep_futures = []
        active_replicas = []
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    f"server '{replica_id}' handle has no get_server_address remote "
                    f"(type={type(handle).__name__}); skipping dynamic endpoint discovery",
                )
                continue
            active_replicas.append(replica_id)
            addr_futures.append(handle.get_server_address.remote())
            ep_futures.append(handle.get_kv_events_endpoints.remote())

        if active_replicas:
            ips_ports = ray.get(addr_futures)
            endpoints_list = ray.get(ep_futures)
            for replica_id, (ip, port), endpoints in zip(active_replicas, ips_ports, endpoints_list, strict=False):
                server_addresses[replica_id] = f"{ip}:{port}"
                if endpoints is None:
                    continue
                kv_event_endpoints[replica_id] = endpoints
        self._manager = CollectorManager(
            self._config.collector,
            collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
        )
        self._manager.start()

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Return construction + routing state for debugging.

        Reports what the balancer was wired with (pool, manager type,
        materialized strategies) and how many routing decisions it has made —
        enough to verify the construction flow over the remote boundary.
        """
        return {
            "servers": list(self._servers.keys()),
            "manager": type(self._manager).__name__,
            "strategies": [{"type": type(s).__name__, "weight": w} for s, w in self._strategies],
            "route_calls": self._route_calls,
            "sticky_size": len(self._sticky),
        }

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes. No-op in v1 (no inflight)."""

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Acquire the best server for a request: delegate to ``route()``, map back.

        Builds ``ReplicaInfo`` candidates from the pool, asks ``route()`` for a
        best-first ranking, and returns ``(ranking[0], handle)``. Raises
        ``RuntimeError`` if no replica is available (empty pool or all blacklisted).

        The ``request_id`` and the sticky-session table are forwarded to
        ``route()`` so strategies can short-circuit to a bound, non-overloaded
        replica. After a ranking is chosen, the binding is refreshed so the
        next turn of the same ``request_id`` stays affinity-bound (or, when a
        sticky replica was overloaded and routing fell back, rebinds to the
        new choice).
        """
        replicas = [ReplicaInfo(replica_id=sid) for sid in self._servers]
        self._route_calls += 1
        ranking = route(
            self._strategies,
            prompt_ids,
            self._store,
            replicas,
            request_id,
            self._sticky,
        )
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        self._sticky.put(request_id, server_id)
        logger.info(
            f"request={request_id} routed to server={server_id} (ranking={ranking}, pool={list(self._servers)})",
        )
        return server_id, self._servers[server_id]

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool.

        Note: the manager is keyed by the endpoint addresses resolved at
        init time, not by this pool, so it is not touched here.
        """
        for sid, handle in servers.items():
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the pool (manager is not keyed by the pool).

        Also invalidates every sticky binding pointing at a removed server so a
        subsequent ``acquire_server`` for a bound conversation doesn't try to
        short-circuit to a dead replica (the strategy would reject it and fall
        back to combined scoring anyway, but clearing early keeps the table
        clean and the logs honest).
        """
        for sid in server_ids:
            self._servers.pop(sid, None)
            self._sticky.invalidate_replica(sid)
