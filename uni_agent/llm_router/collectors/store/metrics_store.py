"""Unified metrics store for reading/writing polling metric data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uni_agent.llm_router.collectors.metric_spec import METRIC_SPECS


@dataclass
class MetricsStore:
    """Unified metrics store: ``{node_id: {canonical_key: value}}``.

    - ``get(node_id, key)``  → single value (falls back to ``METRIC_SPECS[key]["default"]``)
    - ``get(node_id)``       → entire node dict (empty dict if unknown)
    - ``refresh(new_data)``  → batch update; only updates nodes present in ``new_data``,
                               existing nodes NOT in ``new_data`` are left untouched
    """

    _data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, node_id: str, key: str | None = None) -> Any | dict[str, Any]:
        """Read metrics.

        ``get(node_id, key)``  → single value, falls back to spec default or 0
        ``get(node_id)``       → entire node dict
        """
        node = self._data.get(node_id, {})
        if key is not None:
            if key in node:
                return node[key]
            spec = METRIC_SPECS.get(key)
            return spec.get("default", 0) if spec else 0
        return dict(node)

    def refresh(self, new_data: dict[str, dict[str, Any]]) -> None:
        """Batch refresh from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.
        """
        for node_id, metrics in new_data.items():
            existing = self._data.get(node_id, {})
            merged = dict(existing)
            merged.update(metrics)
            self._data[node_id] = merged

    def all_ids(self) -> list[str]:
        """Return all node IDs currently in the store."""
        return list(self._data.keys())
