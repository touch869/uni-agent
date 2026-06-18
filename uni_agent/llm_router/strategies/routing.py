"""route() — weighted replica ranking for the KVCAware router.

The Balancer delegates each request to ``route(strategies, prompt_ids,
provider, replicas)`` and maps ``ranking[0]`` back to a server handle
(detailed_balancer.md §2.3).
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RoutingStrategy(Protocol):
    """Routing scoring strategy.

    Each strategy scores a batch of replicas independently and returns a list
    of the same length and order. ``route()`` weighted-sums the outputs.
    """

    def score(
        self,
        prompt_ids: list[int] | None,
        provider: Any,
        replicas: list[Any],
    ) -> list[float]:
        """Score each replica. Larger is better; negatives are allowed."""
        ...


def _rank_key(score: float) -> float:
    """Sort key treating non-finite scores (NaN/inf) as worst."""
    return score if math.isfinite(score) else float("-inf")


def route(
    strategies: list[tuple[Any, float]],
    prompt_ids: list[int] | None,
    provider: Any,
    replicas: list[Any],
) -> list[str]:
    """Return replica ids ranked best-first.

    Falls back to a random shuffle of replica ids if any strategy raises or
    returns a wrong-length score list — routing remains available even when
    metrics are temporarily unavailable.

    Args:
        strategies: ``[(strategy, weight), ...]`` — weighted strategies.
        prompt_ids: prompt token ids (content-aware routing; may be ``None``).
        provider: ``RouteDataProvider`` for metric queries.
        replicas: ``[ReplicaInfo, ...]`` — candidate replicas.

    Returns:
        Replica ids sorted by total score, best first. Falls back to random
        order on scoring failure.

    Raises:
        RuntimeError: ``replicas`` is empty.
    """
    n = len(replicas)
    if n == 0:
        raise RuntimeError("no available replicas")

    final = [0.0] * n
    for strategy, weight in strategies:
        name = type(strategy).__name__
        try:
            scores = strategy.score(prompt_ids, provider, replicas)
            if len(scores) != n:
                raise ValueError(f"{name}.score() returned {len(scores)} scores, expected {n}")
        except Exception as exc:  # noqa: BLE001
            ids = [r.replica_id for r in replicas]
            random.shuffle(ids)
            logger.warning(
                "route(): %s failed (%s: %s), falling back to random order",
                name, type(exc).__name__, exc,
            )
            return ids
        for idx in range(n):
            final[idx] += weight * scores[idx]

    ranking = sorted(range(n), key=lambda idx: _rank_key(final[idx]), reverse=True)
    logger.info(
        "route(): replicas=%d best=%s score=%.4f",
        n, replicas[ranking[0]].replica_id, final[ranking[0]],
    )
    return [replicas[idx].replica_id for idx in ranking]
