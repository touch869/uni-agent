"""VLLMMetricsDecoder — vLLM Prometheus metrics decoder.

Parses Prometheus exposition-format text and writes results
to MetricsStore.
"""

from __future__ import annotations

import logging
from typing import Any

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.metric_spec import METRIC_SPECS, MetricKey
from uni_agent.llm_router.store.metrics_store import MetricsStore

logger = get_router_logger("vllm-metrics")


def _extract_label(labels_str: str, key: str) -> str | None:
    """Extract a label value from a Prometheus ``k="v",k2="v2"`` labels string."""
    for kv in labels_str.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"')
    return None


class VLLMMetricsDecoder(Decoder):
    """vLLM Prometheus metrics decoder — parses HTTP response text
    and writes results to MetricsStore.

    vLLM Prometheus raw name → canonical key mapping:
        ``vllm:kv_cache_usage_perc``  → ``KV_CACHE_USAGE_PERC``
        ``vllm:num_requests_running`` → ``NUM_REQUESTS_RUNNING``
        ``vllm:num_requests_waiting`` → ``NUM_REQUESTS_WAITING``
    """

    store_cls = MetricsStore

    _PROMETHEUS_MAP: dict[str, str] = {
        "vllm:kv_cache_usage_perc": MetricKey.KV_CACHE_USAGE_PERC,
        "vllm:num_requests_running": MetricKey.NUM_REQUESTS_RUNNING,
        "vllm:num_requests_waiting": MetricKey.NUM_REQUESTS_WAITING,
    }

    def __init__(self) -> None:
        self._store = self.store_cls.default()

    def decode(self, raw_data: bytes | str, node_id: str) -> None:
        """Parse Prometheus text and write results to store.

        Args:
            raw_data: HTTP response text (Prometheus exposition format).
            node_id: Source replica identifier.
        """
        # HTTP delivers str; ignore bytes data
        if isinstance(raw_data, bytes):
            logger.debug("VLLMMetricsDecoder received bytes data, expected str — skipping")
            return

        result: dict[str, Any] = {}
        for line in raw_data.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Parse ``name{labels} value`` — labels needed for cache_config_info.
            try:
                if "{" in line:
                    name_part, rest = line.split("{", 1)
                    labels_str, _, value_part = rest.partition("}")
                    raw_name = name_part.strip()
                else:
                    raw_name = line.split()[0]
                    labels_str, value_part = "", line.split()[-1]
            except (ValueError, IndexError):
                continue
            # cache_config_info is an info gauge: its value is 1.0, the real
            # data lives in the ``num_gpu_blocks`` label.
            if raw_name == "vllm:cache_config_info":
                n = _extract_label(labels_str, "num_gpu_blocks")
                if n is not None:
                    try:
                        result[MetricKey.NUM_GPU_BLOCKS] = int(float(n))
                    except ValueError:
                        pass
                continue
            try:
                value = float(value_part)
            except ValueError:
                continue
            canonical = self._PROMETHEUS_MAP.get(raw_name)
            if canonical:
                value_type = METRIC_SPECS[canonical].get("value_type", float)
                result[canonical] = value_type(value)

        self._store.refresh({node_id: result})
