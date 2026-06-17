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

from uni_agent.llm_router.collectors import RouteDataProvider
from uni_agent.llm_router.config import KVCAwareConfig
from uni_agent.llm_router.strategies import ReplicaInfo, StrategyRegistry, route


class KVCAwareBalancer:
    """Pure-framework router shell. See module docstring."""

    def __init__(self, servers: dict[str, Any], router_config: Any) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        self._config = KVCAwareConfig.from_config(router_config)
        # collection_names = union of every strategy's collector_names.
        collection_names = sorted(
            {name for cfg in self._config.strategies for name in cfg.collector_names}
        )
        # TODO(server_address): RouteDataProvider currently takes only
        # (collectors_config, collection_names); the per-server collection
        # addresses (ip:port) are meant to come from the server handles below,
        # but that injection is not yet implemented in the collectors module.
        # Tracked as a collectors-module design item; provider runs with
        # whatever addresses the collectors_config carries for now.
        self._provider = RouteDataProvider(self._config.collector, collection_names)
        self._provider.start()
        self._strategies: list[tuple[Any, float]] = [
            (StrategyRegistry.get(type(cfg)).from_config(cfg), cfg.weight)
            for cfg in self._config.strategies
        ]
        self._servers: dict[str, Any] = dict(servers)
        self._route_calls = 0

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Return construction + routing state for debugging.

        Reports what the balancer was wired with (pool, provider type,
        materialized strategies) and how many routing decisions it has made —
        enough to verify the construction flow over the remote boundary.
        """
        return {
            "servers": list(self._servers.keys()),
            "provider": type(self._provider).__name__,
            "strategies": [{"type": type(s).__name__, "weight": w} for s, w in self._strategies],
            "route_calls": self._route_calls,
        }

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes. No-op in v1 (no inflight)."""

    def acquire_server(
        self, request_id: str, prompt_ids: list[int] | None = None
    ) -> tuple[str, Any]:
        """Acquire the best server for a request: delegate to ``route()``, map back.

        Builds ``ReplicaInfo`` candidates from the pool, asks ``route()`` for a
        best-first ranking, and returns ``(ranking[0], handle)``. Raises
        ``RuntimeError`` if no replica is available (empty pool or all blacklisted).
        """
        replicas = [ReplicaInfo(replica_id=sid) for sid in self._servers]
        self._route_calls += 1
        ranking = route(self._strategies, prompt_ids, self._provider, replicas)
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        return server_id, self._servers[server_id]

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool.

        Note: the provider is a global collector keyed by configured addresses,
        not by this pool, so it is not touched here (see TODO in ``__init__``).
        """
        for sid, handle in servers.items():
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the pool (provider is not keyed by the pool)."""
        for sid in server_ids:
            self._servers.pop(sid, None)
