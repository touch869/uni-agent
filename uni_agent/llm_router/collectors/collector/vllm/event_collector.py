"""
VLLMKVEventCollector — vLLM KV-cache event collector.
"""

from __future__ import annotations

import msgpack

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.hash import compute_hash
from uni_agent.llm_router.collectors.collector.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.collector.zmq_event_collector import ZMQEventCollector
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("vllm-kv-event-collector")


class VLLMKVEventCollector(ZMQEventCollector):
    """vLLM KV-cache event collector — parses msgpack payloads,
    applies events to ``KVCacheStore``.

    ``KVCacheStore`` is passed in ``start(store)`` — owned by
    ``RouteDataProvider``.  The collector writes to it via CRUD
    methods; it never queries the store.

    Attributes:
        remote_to_local_block_hash: GPU remote block hash to local prefix
            hash, also used to translate CPU BlockStored events.
        cpu_remote_to_local_block_hash: Per-replica CPU remote block hash to
            local prefix hash, retained until CPU BlockRemoved.
    """

    def __init__(self, config, kv_event_addresses: dict[str, list[str]] | None = None) -> None:
        super().__init__(config, kv_event_addresses=kv_event_addresses)
        self.remote_to_local_block_hash: dict[str, str] = {}
        self.cpu_remote_to_local_block_hash: dict[tuple[str, str], str] = {}

    def _consume_payload(self, payload: bytes, node_id: str) -> None:
        """Decode msgpack payload, apply events to store.

        Args:
            payload: Raw ZMQ message bytes.
            node_id: The replica that sent this payload — used as
                     ``default_replica_id`` for ``KVCacheEvent.from_raw``.
        """
        try:
            raw_data = msgpack.unpackb(payload, raw=False)
            events = KVCacheEvent.from_raw(raw_data, default_replica_id=node_id)
            logger.debug(f"consumed payload from {node_id}: {len(events)} events")
            for event in events:
                self._apply_event(event, default_replica_id=node_id)
        except (msgpack.UnpackException, ValueError, TypeError) as e:
            logger.warning(f"failed to parse msgpack payload from {node_id}: {type(e).__name__}: {e}")

    def _apply_event(
        self,
        event: KVCacheEvent,
        default_replica_id: str | None = None,
    ) -> None:
        """Apply a tier-aware vLLM KV-cache event."""
        store = self._store
        replica_id = event.replica_id or default_replica_id or ""
        medium = (event.medium or "gpu").lower()
        seed = 0

        if medium not in {"gpu", "cpu"}:
            logger.warning(
                f"ignoring unsupported KV medium={event.medium!r}: "
                f"replica={replica_id} type={event.event_type}",
            )
            return

        if event.is_store:
            logger.debug(
                f"BlockStored: replica={replica_id}, medium={medium}, "
                f"blocks={len(event.block_hashes)}, block_size={event.block_size}",
            )
            if medium == "cpu":
                local_hashes: list[str] = []
                if event.token_ids is not None:
                    local_parent_hash = seed
                    if event.parent_block_hash is not None:
                        local_parent_str = (
                            self.cpu_remote_to_local_block_hash.get(
                                (replica_id, event.parent_block_hash)
                            )
                            or self.remote_to_local_block_hash.get(
                                event.parent_block_hash
                            )
                        )
                        if local_parent_str is not None:
                            local_parent_hash = int(local_parent_str)

                    for i, block_bytes in enumerate(event.token_ids):
                        if i >= len(event.block_hashes):
                            break
                        local_hash_int = compute_hash(
                            local_parent_hash, block_bytes, seed=seed,
                        )
                        local_hash = str(local_hash_int)
                        remote_hash = event.block_hashes[i]
                        self.cpu_remote_to_local_block_hash[
                            (replica_id, remote_hash)
                        ] = local_hash
                        local_hashes.append(local_hash)
                        local_parent_hash = local_hash_int
                else:
                    for remote_hash in event.block_hashes:
                        local_hash = self.remote_to_local_block_hash.get(remote_hash)
                        if local_hash is None:
                            logger.warning(
                                f"CPU BlockStored has no GPU hash mapping: "
                                f"replica={replica_id} block={remote_hash}",
                            )
                            continue
                        self.cpu_remote_to_local_block_hash[
                            (replica_id, remote_hash)
                        ] = local_hash
                        local_hashes.append(local_hash)
                store.add_blocks(replica_id, local_hashes, tier="cpu")
                return

            if (
                store.block_size is None
                and event.block_size is not None
                and event.block_size > 0
            ):
                store.block_size = event.block_size
            if event.token_ids is None:
                return

            local_parent_hash = seed
            if event.parent_block_hash is not None:
                local_parent_str = self.remote_to_local_block_hash.get(
                    event.parent_block_hash
                )
                if local_parent_str is not None:
                    local_parent_hash = int(local_parent_str)

            local_hashes = []
            for i, block_bytes in enumerate(event.token_ids):
                if i >= len(event.block_hashes):
                    break
                local_hash_int = compute_hash(
                    local_parent_hash, block_bytes, seed=seed,
                )
                local_hash_str = str(local_hash_int)
                remote_hash = event.block_hashes[i]
                self.remote_to_local_block_hash[remote_hash] = local_hash_str
                local_hashes.append(local_hash_str)
                local_parent_hash = local_hash_int
            store.add_blocks(replica_id, local_hashes, tier="gpu")
            return

        if event.is_remove:
            logger.debug(
                f"BlockRemoved: replica={replica_id}, medium={medium}, "
                f"blocks={len(event.block_hashes)}",
            )
            if medium == "cpu":
                local_hashes = [
                    self.cpu_remote_to_local_block_hash[(replica_id, remote_hash)]
                    for remote_hash in event.block_hashes
                    if (replica_id, remote_hash)
                    in self.cpu_remote_to_local_block_hash
                ]
                store.remove_blocks(replica_id, local_hashes, tier="cpu")
                for remote_hash in event.block_hashes:
                    self.cpu_remote_to_local_block_hash.pop(
                        (replica_id, remote_hash), None
                    )
                return

            local_hashes = [
                self.remote_to_local_block_hash[remote_hash]
                for remote_hash in event.block_hashes
                if remote_hash in self.remote_to_local_block_hash
            ]
            store.remove_blocks(replica_id, local_hashes, tier="gpu")
            for remote_hash in event.block_hashes:
                self.remote_to_local_block_hash.pop(remote_hash, None)
            return

        if event.is_clear:
            logger.info(f"AllBlocksCleared: replica={replica_id}")
            store.clear_replica(replica_id)
            # NOTE: only the CPU mapping is per-replica (keyed by
            # (replica_id, remote_hash)) and must be dropped here. The GPU
            # ``remote_to_local_block_hash`` is replica-agnostic and content
            # deterministic (remote_hash → local_hash is a pure function of
            # block content), so it stays valid across a clear — and is shared
            # across replicas, so it must NOT be wiped per-replica.
            stale_cpu_keys = [
                key for key in self.cpu_remote_to_local_block_hash
                if key[0] == replica_id
            ]
            for key in stale_cpu_keys:
                del self.cpu_remote_to_local_block_hash[key]
