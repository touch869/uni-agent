"""MetricsUpdate — structured update command for MetricsStore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricsUpdate:
    """Structured update command for MetricsStore.

    Returned by VLLMMetricsDecoder.decode() — contains metrics data
    to be applied by Collector via DataStore.

    Attributes:
        node_id: Target endpoint identifier.
        metrics: Dict of canonical_key → value.
    """

    node_id: str
    metrics: dict[str, Any]
