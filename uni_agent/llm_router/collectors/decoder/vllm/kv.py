"""VLLMKVDecoder — vLLM KV-cache event decoder.

Decodes msgpack payloads from ZMQ, applies KV cache events to
KVCacheStore via dispatch table.
"""

from __future__ import annotations

from collections import defaultdict

import msgpack

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.decoder.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.hash import compute_hash
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.store.kv_cache_store import KVCacheStore

logger = get_router_logger("vllm-kv")

# Log a kv-events tally (events + blocks by type) every N applied events.
# Lets us see BlockStored/BlockRemoved flow — esp. whether mc-off groups (no
# mooncake → no kv-events emission) get any events at all.
_KV_EVENT_LOG_EVERY = 500


class VLLMKVDecoder(Decoder):
    """vLLM KV-cache decoder — msgpack payload → KVCacheStore updates.

    Event dispatch uses a ``_DISPATCH`` table mapping ``event_type``
    to handler methods — no if/else chain needed.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  Used in
            _handle_stored for chained hash computation.
    """

    store_cls = KVCacheStore

    # event_type → handler method name
    _DISPATCH: dict[str, str] = {
        "stored": "_handle_stored",
        "removed": "_handle_removed",
        "clear": "_handle_clear",
    }

    def __init__(self) -> None:
        self._store = self.store_cls.default()
        self.remote_to_local_block_hash: dict[str, str] = {}
        # kv-event tallies for periodic summary logging.
        self._event_counts: dict[str, int] = defaultdict(int)
        self._block_counts: dict[str, int] = defaultdict(int)
        self._last_logged_total = 0

    def decode(self, raw_data: bytes | str, node_id: str) -> None:
        """Decode msgpack payload, apply events to store.

        Args:
            raw_data: ZMQ payload bytes (msgpack-encoded).
            node_id: The replica that sent this payload.
        """
        # ZMQ delivers bytes; ignore string data (shouldn't happen for this decoder)
        if isinstance(raw_data, str):
            logger.debug("VLLMKVDecoder received string data, expected bytes — skipping")
            return

        try:
            raw = msgpack.unpackb(raw_data, raw=False)
            events = KVCacheEvent.from_raw(raw, default_replica_id=node_id)
            for event in events:
                self._apply_event(event, default_replica_id=node_id)
        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            logger.warning(f"Failed to decode msgpack payload from node {node_id}: {exc}")

    def _apply_event(
        self,
        event: KVCacheEvent,
        default_replica_id: str | None = None,
    ) -> None:
        """Dispatch a KVCacheEvent to the appropriate handler."""
        handler_name = self._DISPATCH.get(event.event_type)
        if handler_name is None:
            logger.debug(f"Unhandled event type: {event.event_type}")
            return
        handler = getattr(self, handler_name)
        replica_id = event.replica_id or default_replica_id or ""
        n_blocks = len(getattr(event, "block_hashes", []) or [])
        logger.debug(f"kv-event type={event.event_type} replica={replica_id} n_blocks={n_blocks}")
        handler(event, replica_id)

        # Tally for periodic summary — observe BlockStored/BlockRemoved flow.
        self._event_counts[event.event_type] += 1
        self._block_counts[event.event_type] += n_blocks
        total = sum(self._event_counts.values())
        if total - self._last_logged_total >= _KV_EVENT_LOG_EVERY:
            self._last_logged_total = total
            retained = self._store.per_replica_block_counts()
            logger.info(
                f"kv-events tally: events={dict(self._event_counts)} blocks={dict(self._block_counts)} "
                f"(total_events={total}) | retained_blocks/replica={retained}"
            )

    # ── Event handlers ──────────────────────────────────────────────────

    def _handle_stored(self, event: KVCacheEvent, replica_id: str) -> None:
        """Handle BlockStored: learn block_size, compute local hashes, update store."""
        store = self._store
        seed = 0

        if store.block_size is None and event.block_size is not None:
            store.block_size = event.block_size

        if event.token_ids is not None:
            local_parent_hash = seed
            if event.parent_block_hash is not None:
                local_parent_str = self.remote_to_local_block_hash.get(event.parent_block_hash)
                if local_parent_str is not None:
                    local_parent_hash = int(local_parent_str)

            local_hashes: list[str] = []
            for i, block_bytes in enumerate(event.token_ids):
                if i >= len(event.block_hashes):
                    break
                local_hash_int = compute_hash(
                    local_parent_hash,
                    block_bytes,
                    seed=seed,
                )
                local_hash_str = str(local_hash_int)
                bh = event.block_hashes[i]
                self.remote_to_local_block_hash[bh] = local_hash_str
                local_hashes.append(local_hash_str)
                local_parent_hash = local_hash_int  # chain

            store.add_blocks(replica_id, local_hashes)

    def _handle_removed(self, event: KVCacheEvent, replica_id: str) -> None:
        """Handle BlockRemoved: convert remote hashes to local, remove from store."""
        store = self._store
        local_hashes = [
            self.remote_to_local_block_hash[bh] for bh in event.block_hashes if bh in self.remote_to_local_block_hash
        ]
        store.remove_blocks(replica_id, local_hashes)
        for bh in event.block_hashes:
            self.remote_to_local_block_hash.pop(bh, None)

    def _handle_clear(self, event: KVCacheEvent, replica_id: str) -> None:
        """Handle AllBlocksCleared: clear all blocks for the replica."""
        self._store.clear_replica(replica_id)
