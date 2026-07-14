"""StickyDecoder — Balancer callback → StickyUpdate (per-request LRU writes).

Fed by ``CallbackTransport``, which packs each Balancer callback into a
``StatisticEvent``. The decoder is stateless — it dispatches on the event
type and emits a ``StickyUpdate``; the ``Collector`` applies it to the
``StickySessionStore`` via ``DataStore``.

Event → action mapping:
- ``on_acquire(request_id, replica_id)``     → ``put`` (bind/refresh)
- ``on_servers_removed(server_ids)``          → ``invalidate_replica`` (bulk clear)
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.collectors.decoder import Decoder, StickyUpdate
from uni_agent.llm_router.collectors.transport.callback import StatisticEvent
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("sticky-decoder")


class StickyDecoder(Decoder):
    """Decode ``StatisticEvent`` → ``StickyUpdate`` for the sticky-session store."""

    def decode(self, raw_data: bytes | str | Any, node_id: str) -> StickyUpdate | None:
        """Dispatch on the event type; ignore non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.event == "on_acquire":
            if event.request_id is None or event.replica_id is None:
                logger.debug("on_acquire event missing request_id/replica_id — skipping")
                return None
            return StickyUpdate(action="put", request_id=event.request_id, replica_id=event.replica_id)

        if event.event == "on_servers_removed":
            return StickyUpdate(action="invalidate_replica", replica_ids=tuple(event.server_ids))

        # on_release has no sticky meaning — sticky only writes on acquire/remove.
        return None
