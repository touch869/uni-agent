"""
VLLMKVEventCollector — vLLM KV-cache event collector.
"""

from __future__ import annotations

import logging

import msgpack

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.hash import compute_hash
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.collector.zmq_event_collector import ZMQEventCollector
from uni_agent.llm_router.collectors.store.kv_cache_store import KVCacheStore

logger = logging.getLogger(__name__)


class VLLMKVEventCollector(ZMQEventCollector):
    """vLLM KV-cache event collector — parses msgpack payloads,
    applies events to KVCacheStore.

    The store is created from store_cls.default() in __init__ and
    shared across all collectors of the same type (singleton).
    The collector writes to it via CRUD methods; it never queries the store.

    Event dispatch uses a ``_DISPATCH`` table mapping ``event_type``
    to handler methods — no if/else chain needed.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  This is used in
            _handle_stored for chained hash computation.
    """

    store_cls = KVCacheStore

    # event_type → handler method name
    _DISPATCH: dict[str, str] = {
        "stored": "_handle_stored",
        "removed": "_handle_removed",
        "clear": "_handle_clear",
    }

    def __init__(self, config, kv_event_addresses: dict[str, list[str]] | None = None) -> None:
        super().__init__(config, kv_event_addresses=kv_event_addresses)
        self._store = self.store_cls.default()
        self.remote_to_local_block_hash: dict[str, str] = {}

    def _consume_payload(self, payload: bytes, node_id: str) -> None:
        """Decode msgpack payload, apply events to store.

        Args:
            payload: Raw ZMQ message bytes.
            node_id: The replica that sent this payload — used as
                     default_replica_id for KVCacheEvent.from_raw.
        """
        try:
            raw_data = msgpack.unpackb(payload, raw=False)
            events = KVCacheEvent.from_raw(raw_data, default_replica_id=node_id)
            for event in events:
                self._apply_event(event, default_replica_id=node_id)
        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            logger.warning(
                f"Failed to decode msgpack payload from node {node_id}: {exc}"
            )

    def _apply_event(
        self,
        event: KVCacheEvent,
        default_replica_id: str | None = None,
    ) -> None:
        """Dispatch a KVCacheEvent to the appropriate handler via _DISPATCH table."""
        handler_name = self._DISPATCH.get(event.event_type)
        if handler_name is None:
            logger.debug("Unhandled event type: %s", event.event_type)
            return
        handler = getattr(self, handler_name)
        handler(event, default_replica_id)

    # ── Event handlers ──────────────────────────────────────────────────

    def _handle_stored(self, event: KVCacheEvent, replica_id: str) -> None:
        """Handle BlockStored: learn block_size, compute local hashes, update store."""
        store = self._store
        seed = 0

        if store.block_size is None and event.block_size is not None:
            store.block_size = event.block_size

        if event.token_ids is not None:
            # Determine local_parent_hash for chained computation
            local_parent_hash = seed
            if event.parent_block_hash is not None:
                local_parent_str = self.remote_to_local_block_hash.get(
                    event.parent_block_hash
                )
                if local_parent_str is not None:
                    local_parent_hash = int(local_parent_str)

            local_hashes: list[str] = []
            for i, block_bytes in enumerate(event.token_ids):
                if i >= len(event.block_hashes):
                    break
                local_hash_int = compute_hash(
                    local_parent_hash, block_bytes, seed=seed,
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
            self.remote_to_local_block_hash[bh]
            for bh in event.block_hashes
            if bh in self.remote_to_local_block_hash
        ]
        store.remove_blocks(replica_id, local_hashes)
        for bh in event.block_hashes:
            self.remote_to_local_block_hash.pop(bh, None)

    def _handle_clear(self, event: KVCacheEvent, replica_id: str) -> None:
        """Handle AllBlocksCleared: clear all blocks for the replica."""
        self._store.clear_replica(replica_id)
