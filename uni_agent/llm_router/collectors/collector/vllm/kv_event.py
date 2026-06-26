"""
KVCacheEvent — standardized KV cache event data structure.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


def _normalize_block_hash(block_hash: Any) -> str:
    """Normalize vLLM event block hashes across GPU and CPU events.

    vLLM 0.18 GPU KV events apply ``maybe_convert_block_hash`` and, by
    default, publish the low 64 bits of the internal SHA256 block hash as an
    int. Native CPU offload events publish the raw internal ``bytes`` hash.
    Convert raw bytes the same way so both event streams share one key space.
    """
    if isinstance(block_hash, (bytes, bytearray)):
        return str(int.from_bytes(block_hash, byteorder="big") & ((1 << 64) - 1))
    return str(block_hash)


@dataclass(frozen=True)
class KVCacheEvent:
    """Standardized KV cache event — normalized from backend-specific ZMQ payloads.

    Attributes:
        event_type: ``"stored"`` / ``"removed"`` / ``"clear"``.
        replica_id: Source replica (from ZMQ connection or group_idx).
        block_hashes: Block hashes involved in the event (list from msgpack).
        parent_block_hash: Parent block hash — single value shared by all
                           block_hashes in a BlockStored event.
        token_ids: Pre-chopped block token bytes (``list[bytes]``, only present
                   in BlockStored events).  Each element is one full block
                   encoded as uint32 big-endian (4 bytes per token).
        block_size: Block size (only present in BlockStored events).
        medium: Cache medium normalized to lowercase (for example ``"gpu"``
                or ``"cpu"``), when provided by the backend.
    """

    event_type: str
    replica_id: str
    block_hashes: list[str]
    parent_block_hash: str | None
    token_ids: list[bytes] | None
    block_size: int | None
    medium: str | None = None

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw_data: Any, default_replica_id: str | None = None) -> list[KVCacheEvent]:
        """Parse msgpack-decoded raw data into a list of KVCacheEvent instances.

        Parsing pattern::

            timestamp    = raw_data[0]               # ignored for routing
            event_list   = raw_data[1]               # list of events
            event_tag    = raw_data[1][i][0]          # tag for event i
            event_fields = raw_data[1][i][1:]         # fields for event i

        Args:
            raw_data: Msgpack-decoded list ``[timestamp, [[tag, fields...], ...]]``.
            default_replica_id: Fallback replica_id when raw data lacks it.

        Returns:
            A list of KVCacheEvent instances.  Malformed events are skipped.

        Raises:
            ValueError: If the top-level format is invalid.
        """
        if not isinstance(raw_data, (list, tuple)) or len(raw_data) < 2:
            raise ValueError(f"Expected list with >= 2 elements, got {type(raw_data)}")

        event_list = raw_data[1]
        if not isinstance(event_list, (list, tuple)):
            raise ValueError(f"Expected raw_data[1] as event list, got {type(event_list)}")

        results: list[KVCacheEvent] = []
        for event_entry in event_list:
            if not isinstance(event_entry, (list, tuple)) or len(event_entry) < 1:
                continue

            tag = event_entry[0]
            fields = event_entry[1:]
            event_type = cls._resolve_event_type(tag)
            if event_type.startswith("unknown"):
                continue

            replica_id = default_replica_id or ""
            try:
                event = cls._build_event(event_type, fields, replica_id)
                if event is not None:
                    results.append(event)
            except (ValueError, TypeError, IndexError):
                continue

        return results

    # ── Build helpers ────────────────────────────────────────────────────

    @classmethod
    def _build_event(
        cls, event_type: str, fields: list | tuple, replica_id: str,
    ) -> KVCacheEvent | None:
        """Dispatch to the appropriate builder by event_type."""
        if event_type == "stored":
            return cls._build_block_stored(fields, replica_id)
        elif event_type == "removed":
            return cls._build_block_removed(fields, replica_id)
        elif event_type == "clear":
            return cls._build_all_blocks_cleared(replica_id)
        return None

    @classmethod
    def _build_block_stored(cls, fields: list | tuple, replica_id: str) -> KVCacheEvent:
        """Build a BlockStored event from its field list.

        Field order (after tag):
            0: block_hashes         — list of block hash values
            1: parent_block_hash    — single value or None
            2: token_ids            — list of int or None
            3: block_size           — int
            4: lora_id              — deprecated, ignored
            5: medium               — str or None

        Raw ``token_ids`` (``list[int]``) are chopped into full blocks
        and encoded as uint32 big-endian bytes via ``_convert_token_ids``.
        """
        if len(fields) < 4:
            raise ValueError(f"BlockStored needs >= 4 fields, got {len(fields)}")

        block_hashes = [_normalize_block_hash(bh) for bh in fields[0]]
        parent_block_hash = (
            _normalize_block_hash(fields[1]) if fields[1] is not None else None
        )
        raw_token_ids = list(fields[2]) if fields[2] is not None else None
        block_size = int(fields[3])
        medium = (
            str(fields[5]).lower()
            if len(fields) > 5 and fields[5] is not None
            else None
        )

        # Chop and encode token IDs into block-sized uint32 big-endian bytes
        # only for regular GPU events. Native CPU-offload events intentionally
        # carry token_ids=[] and block_size=0 because their block hashes already
        # identify the offloaded entries.
        token_ids = (
            _convert_token_ids(raw_token_ids, block_size)
            if raw_token_ids and block_size > 0
            else None
        )

        return cls(
            event_type="stored",
            replica_id=replica_id,
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            medium=medium,
        )

    @classmethod
    def _build_block_removed(cls, fields: list | tuple, replica_id: str) -> KVCacheEvent:
        """Build a BlockRemoved event from its field list.

        Field order (after tag):
            0: block_hashes — list of block hash values
            1: medium       — str or None
            2: group_idx    — int or None
        """
        block_hashes = [_normalize_block_hash(bh) for bh in fields[0]]
        medium = (
            str(fields[1]).lower()
            if len(fields) > 1 and fields[1] is not None
            else None
        )

        return cls(
            event_type="removed",
            replica_id=replica_id,
            block_hashes=block_hashes,
            parent_block_hash=None,
            token_ids=None,
            block_size=None,
            medium=medium,
        )

    @classmethod
    def _build_all_blocks_cleared(cls, replica_id: str) -> KVCacheEvent:
        """Build an AllBlocksCleared event — no fields after tag."""
        return cls(
            event_type="clear",
            replica_id=replica_id,
            block_hashes=[],
            parent_block_hash=None,
            token_ids=None,
            block_size=None,
            medium=None,
        )

    # ── Tag resolution ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_event_type(tag: Any) -> str:
        """Map msgspec struct tag to canonical event type string.

        vLLM msgspec uses numeric tags:
          - 0 → BlockStored
          - 1 → BlockRemoved
          - 2 → AllBlocksCleared

        String tags are also supported.
        """
        if isinstance(tag, int):
            return {0: "stored", 1: "removed", 2: "clear"}.get(tag, f"unknown_{tag}")

        tag_str = str(tag).lower()
        if "stored" in tag_str:
            return "stored"
        elif "removed" in tag_str or "evicted" in tag_str:
            return "removed"
        elif "clear" in tag_str:
            return "clear"
        return f"unknown_{tag}"

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def is_store(self) -> bool:
        """True if this is a block-stored event."""
        return "stored" in self.event_type.lower()

    @property
    def is_remove(self) -> bool:
        """True if this is a block-removed event."""
        return any(k in self.event_type.lower() for k in ("removed", "evicted"))

    @property
    def is_clear(self) -> bool:
        """True if this is an all-blocks-cleared event."""
        return "clear" in self.event_type.lower()


# ── Token ID conversion ────────────────────────────────────────────────────


def _convert_token_ids(raw_ids: list[int], block_size: int) -> list[bytes]:
    """Convert raw token IDs to list of block-sized uint32 big-endian byte chunks.

    Each block of token IDs is encoded as uint32 big-endian (4 bytes per token),
    matching aibrix's ``convertTokenIDs`` / ``tokenIDsToBytes``.

    Args:
        raw_ids: Raw token IDs as ``list[int]``.
        block_size: Number of tokens per block (must be > 0).

    Returns:
        List of bytes objects, one per full block.

    Raises:
        ValueError: If ``block_size <= 0`` or ``len(raw_ids)`` is not
            divisible by ``block_size``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if len(raw_ids) % block_size != 0:
        raise ValueError(
            f"token_ids len={len(raw_ids)} not divisible by block_size={block_size}"
        )
    num_blocks = len(raw_ids) // block_size
    result: list[bytes] = []
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        result.append(struct.pack(f">{block_size}I", *raw_ids[start:end]))
    return result
