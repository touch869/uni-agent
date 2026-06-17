"""route() — weighted replica ranking entry (deferred).

The Balancer delegates each request to ``route(strategies, prompt_ids,
provider, replicas)`` and maps ``ranking[0]`` back to a server handle
(detailed_balancer.md §2.3). The ranking algorithm — weighted scoring across
strategies, the per-request KV-cache prefix-hit query, blacklist at
load >= threshold — is the strategy-module detailed design's responsibility;
this entry is a placeholder until that lands.
"""

from __future__ import annotations

from typing import Any


def route(
    strategies: list[tuple[Any, float]],
    prompt_ids: list[int] | None,
    provider: Any,
    replicas: list[Any],
) -> list[str]:
    """Return replica ids ranked best-first (deferred).

    Args:
        strategies: ``[(strategy, weight), ...]`` — weighted strategies.
        prompt_ids: prompt token ids (content-aware routing; may be ``None``).
        provider: ``RouteDataProvider`` for metric queries.
        replicas: ``[ReplicaInfo, ...]`` — candidate replicas.

    Returns:
        Replica ids ranked best-first.
    """
    # Placeholder ranking: replica ids in input order (no scoring). Enough to
    # exercise the Balancer's wiring end-to-end; the real KV-aware ranking —
    # weighted scoring, per-request KV-cache prefix-hit query, blacklist at
    # load >= threshold — is deferred to the strategy-module detailed design.
    return [r.replica_id for r in replicas]
