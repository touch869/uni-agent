"""vLLM Prometheus polling collector."""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.config.router import CollectorConfig
from uni_agent.llm_router.collectors.metric_spec import METRIC_SPECS, MetricKey
from uni_agent.llm_router.collectors.collector.polling_collector import PollingCollector
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("vllm-polling-collector")


class VLLMPollingCollector(PollingCollector):
    """vLLM backend polling collector."""

    # vLLM Prometheus raw name → canonical key mapping
    _PROMETHEUS_MAP: dict[str, str] = {
        "vllm:kv_cache_usage_perc":  MetricKey.KV_CACHE_USAGE_PERC,
        "vllm:num_requests_running": MetricKey.NUM_REQUESTS_RUNNING,
        "vllm:num_requests_waiting": MetricKey.NUM_REQUESTS_WAITING,
    }

    def __init__(self, config, server_addresses: dict[str, str] | None = None) -> None:
        super().__init__(config, server_addresses=server_addresses)

    def _parse_response(self, text: str, node_id: str) -> dict[str, Any]:
        """Parse vLLM Prometheus exposition-format text into {canonical_key: value} dict."""
        result: dict[str, Any] = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            raw_name = line.split("{")[0] if "{" in line else line.split()[0]
            value = float(line.split()[-1])
            canonical = self._PROMETHEUS_MAP.get(raw_name)
            if canonical:
                value_type = METRIC_SPECS[canonical].get("value_type", float)
                result[canonical] = value_type(value)
        logger.debug(f"VLLMPollingCollector parsed: node={node_id} metrics={result}")
        return result
