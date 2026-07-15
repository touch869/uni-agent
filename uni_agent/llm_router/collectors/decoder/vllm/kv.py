"""VLLMKVDecoder — vLLM KV-cache event decoder.

Decodes msgpack payloads from ZMQ and returns structured update commands.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

import msgpack

from uni_agent.llm_router.collectors.decoder import Decoder, KVCacheUpdate
from uni_agent.llm_router.collectors.decoder.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.types import Layer
from uni_agent.llm_router.utils.hash import compute_hash

logger = get_router_logger("vllm-kv")


class VLLMKVDecoder(Decoder):
    """vLLM KV-cache decoder — msgpack payload → KVCacheUpdate.

    Each event type has a dedicated handler that folds its blocks into a single
    ``KVCacheUpdate`` accumulator; ``decode`` is pure dispatch.

    Attributes:
        vllm_to_local_block_hash: Mapping keyed by cache layer, node ID, and
            vLLM block hash, with the locally-computed block hash as its
            value. Layer and node scoping prevents cross-layer/cross-node removal.
        _block_size: Learned block size from the first BlockStored event.
    """

    _SUPPORTED_EVENT_LAYERS = {Layer.GPU, Layer.CPU}

    def __init__(self) -> None:
        self.vllm_to_local_block_hash: dict[tuple[Layer, str, str], str] = {}
        self._block_size: int | None = None

    @staticmethod
    def _hash_mapping_key(layer: Layer, node_id: str, vllm_block_hash: str) -> tuple[Layer, str, str]:
        return (layer, node_id, vllm_block_hash)

    def _lookup_local_hash(self, layer: Layer, node_id: str, vllm_block_hash: str) -> str | None:
        return self.vllm_to_local_block_hash.get(self._hash_mapping_key(layer, node_id, vllm_block_hash))

    def _find_local_hash(
        self,
        node_id: str,
        vllm_block_hash: str,
        preferred_layer: Layer,
    ) -> str | None:
        local_hash = self._lookup_local_hash(preferred_layer, node_id, vllm_block_hash)
        if local_hash is not None:
            return local_hash

        for layer in Layer:
            if layer != preferred_layer and layer in self._SUPPORTED_EVENT_LAYERS:
                local_hash = self._lookup_local_hash(layer, node_id, vllm_block_hash)
                if local_hash is not None:
                    return local_hash
        return None

    def _record_local_hash_mapping(
        self,
        layer: Layer,
        node_id: str,
        vllm_block_hash: str,
        local_hash: str,
    ) -> None:
        self.vllm_to_local_block_hash[self._hash_mapping_key(layer, node_id, vllm_block_hash)] = local_hash

    def _pop_local_hash_mapping(self, layer: Layer, node_id: str, vllm_block_hash: str) -> str | None:
        return self.vllm_to_local_block_hash.pop(self._hash_mapping_key(layer, node_id, vllm_block_hash), None)

    def _clear_local_hash_mappings(self, node_id: str) -> None:
        stale_keys = [key for key in self.vllm_to_local_block_hash if key[1] == node_id]
        for key in stale_keys:
            del self.vllm_to_local_block_hash[key]

    @classmethod
    def _medium_to_layer(cls, medium: str | None) -> Layer | None:
        """Map a vLLM medium to a supported canonical layer.

        Missing medium values from older vLLM versions default to GPU.
        Unsupported non-empty values return ``None``.
        """
        if medium is None:
            return Layer.GPU
        try:
            layer = Layer(medium.lower())
        except ValueError:
            return None
        return layer if layer in cls._SUPPORTED_EVENT_LAYERS else None

    def decode(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | None:
        """Decode msgpack payload and return a structured update command.

        Handles both single event (real-time) and multiple events (replay):
          - Single: [timestamp, [[tag, fields...], ...]]
          - Multiple: [[timestamp, [...]], [timestamp, [...]]]

        Args:
            raw_data: ZMQ payload bytes (msgpack-encoded).
            node_id: The endpoint that sent this payload.

        Returns:
            KVCacheUpdate with operations to apply, or None if decode failed.
        """
        if isinstance(raw_data, str):
            logger.debug("VLLMKVDecoder received string data, expected bytes — skipping")
            return None

        try:
            raw = msgpack.unpackb(raw_data, raw=False)
            if not isinstance(raw, list) or not raw:
                logger.warning("Unexpected msgpack format from node %s", node_id)
                return None
            event_payloads = raw if isinstance(raw[0], list) else [raw]

            update = KVCacheUpdate(node_id=node_id)
            for payload in event_payloads:
                events = KVCacheEvent.from_raw(payload, default_node_id=node_id)
                for event in events:
                    if event.event_type == "stored":
                        self._on_block_stored(event, update)
                    elif event.event_type == "removed":
                        self._on_block_removed(event, update)
                    elif event.event_type == "clear":
                        self._on_all_blocks_cleared(event, update)
                    else:
                        raise ValueError(f"Unknown event.event_type {event.event_type}.")
            return update

        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to decode msgpack payload from node %s: %s",
                node_id,
                exc,
            )
            return None

    # ── Event handlers ──────────────────────────────────────────────────

    def _on_block_stored(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Compute or translate stored hashes and add them to their layer."""
        layer = self._medium_to_layer(event.medium)
        if layer is None:
            logger.warning(
                "Ignoring unsupported KV medium=%r from node %s",
                event.medium,
                update.node_id,
            )
            return

        if self._can_compute_local_hashes(event, update.node_id, layer):
            local_hashes = self._compute_local_hashes(event, update.node_id, layer)
            if self._block_size is None and event.block_size is not None and event.block_size > 0:
                self._block_size = event.block_size
                update.set_block_size(event.block_size)
        else:
            local_hashes = self._resolve_local_hashes(event, update.node_id, layer)

        update.add(layer, local_hashes)

    def _can_compute_local_hashes(
        self,
        event: KVCacheEvent,
        node_id: str,
        layer: Layer,
    ) -> bool:
        if not event.token_ids or event.block_size is None or event.block_size <= 0:
            return False
        if len(event.token_ids) != len(event.block_hashes):
            logger.warning(
                "BlockStored hash/token count mismatch: node=%s hashes=%d token_blocks=%d",
                node_id,
                len(event.block_hashes),
                len(event.token_ids),
            )
            return False
        if event.parent_block_hash is None:
            return True
        return self._find_local_hash(
            node_id,
            event.parent_block_hash,
            preferred_layer=layer,
        ) is not None

    def _compute_local_hashes(
        self,
        event: KVCacheEvent,
        node_id: str,
        layer: Layer,
    ) -> list[str]:
        """Compute chained local hashes for a GPU or CPU stored event."""
        local_parent_hash = 0
        if event.parent_block_hash is not None:
            local_parent_str = self._find_local_hash(
                node_id,
                event.parent_block_hash,
                preferred_layer=layer,
            )
            if local_parent_str is not None:
                local_parent_hash = int(local_parent_str)

        local_hashes: list[str] = []
        for vllm_block_hash, block_bytes in zip(event.block_hashes, event.token_ids or [], strict=True):
            local_hash_int = compute_hash(local_parent_hash, block_bytes, seed=0)
            local_hash = str(local_hash_int)
            self._record_local_hash_mapping(layer, node_id, vllm_block_hash, local_hash)
            local_hashes.append(local_hash)
            local_parent_hash = local_hash_int
        return local_hashes

    def _resolve_local_hashes(
        self,
        event: KVCacheEvent,
        node_id: str,
        target_layer: Layer,
    ) -> list[str]:
        """Resolve token-less stored events through existing hash mappings."""
        local_hashes: list[str] = []
        for vllm_block_hash in event.block_hashes:
            local_hash = self._find_local_hash(
                node_id,
                vllm_block_hash,
                preferred_layer=target_layer,
            )
            if local_hash is None:
                logger.warning(
                    "BlockStored has no known hash mapping: node=%s layer=%s vllm_block=%s",
                    node_id,
                    target_layer.value,
                    vllm_block_hash,
                )
                continue
            self._record_local_hash_mapping(target_layer, node_id, vllm_block_hash, local_hash)
            local_hashes.append(local_hash)
        return local_hashes

    def _on_block_removed(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Remove hashes only from the layer named by the event."""
        layer = self._medium_to_layer(event.medium)
        if layer is None:
            logger.warning(
                "Ignoring unsupported KV medium=%r from node %s",
                event.medium,
                update.node_id,
            )
            return

        local_hashes: list[str] = []
        for vllm_block_hash in event.block_hashes:
            local_hash = self._pop_local_hash_mapping(layer, update.node_id, vllm_block_hash)
            if local_hash is not None:
                local_hashes.append(local_hash)
        update.remove(layer, local_hashes)

    def _on_all_blocks_cleared(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Clear both store state and decoder mappings for one node."""
        self._clear_local_hash_mappings(update.node_id)
        update.clear()
