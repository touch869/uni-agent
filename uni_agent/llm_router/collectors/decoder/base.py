"""Decoder abstract base and its update return types.

A ``Decoder`` turns a raw transport payload (bytes/str) into a structured
update that the ``Collector`` applies to the store.  The update types
(``KVCacheUpdate``, ``MetricsUpdate``) live here, next to the base, because
they are the decoder's output contract — the layer that both produces and
consumes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from uni_agent.llm_router.types import Layer


@dataclass
class KVCacheUpdate:
    """Mutable accumulator for KVCacheStore updates, built by the decoder.

    The decoder dispatches each event to a handler that calls ``add``/``remove``
    to fold blocks in; the collector then reads ``add_blocks``/``remove_blocks``
    to write the store.

    Attributes:
        node_id: Target endpoint identifier.
        add_blocks: layer → block hashes accumulated to add.
        remove_blocks: layer → block hashes accumulated to remove.
        clear_all: If True, clear all blocks for this node.
        block_size: Block size learned from the first BlockStored (None until then).
    """

    node_id: str
    add_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    remove_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    clear_all: bool = False
    block_size: int | None = None

    def add(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold stored blocks into ``add_blocks`` under ``layer``."""
        self.add_blocks.setdefault(layer, []).extend(block_hashes)

    def remove(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold removed blocks into ``remove_blocks`` under ``layer``."""
        self.remove_blocks.setdefault(layer, []).extend(block_hashes)

    def clear(self) -> None:
        """Mark the replica for a full block clear."""
        self.clear_all = True

    def set_block_size(self, size: int) -> None:
        """Set the learned block size (first BlockStored wins; set by the decoder)."""
        self.block_size = size


@dataclass
class MetricsUpdate:
    """Structured update command for MetricsStore.

    Attributes:
        node_id: Target endpoint identifier.
        metrics: Dict of canonical_key → value.
    """

    node_id: str
    metrics: dict[str, Any]


class Decoder(ABC):
    """Abstract base for data decoders.

    Subclasses implement ``decode()`` with their backend-specific parsing logic,
    returning a ``KVCacheUpdate`` or ``MetricsUpdate`` (or ``None`` on failure).
    """

    @abstractmethod
    def decode(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | MetricsUpdate | None:
        """Decode raw data and return a structured update.

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) or ``str``
                (from HTTP response text).
            node_id: Source endpoint/node identifier.

        Returns:
            A ``KVCacheUpdate`` or ``MetricsUpdate``, or ``None`` if decode fails.
        """
