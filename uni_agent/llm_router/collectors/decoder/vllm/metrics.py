"""VLLMMetricsDecoder — vLLM Prometheus metrics decoder.

Parses Prometheus exposition-format text and writes results
to MetricsStore.
"""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.metric_spec import METRIC_SPECS, MetricKey
from uni_agent.llm_router.store.metrics_store import MetricsStore

logger = get_router_logger("vllm-metrics")

# Log polled Prometheus metrics every N decode calls (≈every 10 s at the
# default 1 s polling interval × a few replicas). Lets us compare what the
# collector feeds the router against vllm's own engine-stats log.
_METRICS_LOG_EVERY_POLLS = 30

# Cumulative metrics tracked for windowed deltas in the evidence log. Single
# source of truth — ``_delta`` consumers below read these by key, and the
# per-replica prev-snapshot iterates the same tuple.
_CUMULATIVE_KEYS: tuple[str, ...] = (
    MetricKey.TTFT_SECONDS_SUM, MetricKey.TTFT_COUNT,
    MetricKey.QUEUE_TIME_SECONDS_SUM, MetricKey.QUEUE_TIME_COUNT,
    MetricKey.TPOT_SECONDS_SUM, MetricKey.TPOT_COUNT,
    MetricKey.PROMPT_TOKENS, MetricKey.PROMPT_TOKENS_CACHED,
    MetricKey.GENERATION_TOKENS, MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
)


def _avg(delta_sum: float, delta_cnt: float) -> float:
    """Windowed average = delta_sum / delta_cnt, or NaN if no samples."""
    return delta_sum / delta_cnt if delta_cnt > 0 else float("nan")


def _ms(value: float) -> str:
    """Format a seconds value as millis for the evidence log ('-' if NaN)."""
    return f"{value * 1000:.1f}" if value == value else "-"


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
        # vLLM 0.21 Prometheus counters carry a ``_total`` suffix (confirmed by
        # scraping /metrics); the pre-Task-2 names without ``_total`` never matched.
        "vllm:prefix_cache_queries_total": MetricKey.PREFIX_CACHE_QUERIES,
        "vllm:prefix_cache_hits_total": MetricKey.PREFIX_CACHE_HITS,
        # Evidence metrics: TTFT/TPOT histograms + token/external counters.
        # All cumulative — the periodic log emits windowed deltas (rates/averages).
        "vllm:time_to_first_token_seconds_sum": MetricKey.TTFT_SECONDS_SUM,
        "vllm:time_to_first_token_seconds_count": MetricKey.TTFT_COUNT,
        # Queue wait (TTFT includes it; prefill_time = TTFT - queue is the real
        # prefill cost that prefix-sharing reduces).
        "vllm:request_queue_time_seconds_sum": MetricKey.QUEUE_TIME_SECONDS_SUM,
        "vllm:request_queue_time_seconds_count": MetricKey.QUEUE_TIME_COUNT,
        # NOTE: TPOT histogram is ``request_time_per_output_token_seconds`` in
        # vLLM 0.21 (NOT ``time_per_output_token_seconds``).
        "vllm:request_time_per_output_token_seconds_sum": MetricKey.TPOT_SECONDS_SUM,
        "vllm:request_time_per_output_token_seconds_count": MetricKey.TPOT_COUNT,
        "vllm:generation_tokens_total": MetricKey.GENERATION_TOKENS,
        "vllm:external_prefix_cache_hits_total": MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
        # PROMPT_TOKENS / PROMPT_TOKENS_CACHED are NOT here — they come from the
        # labeled ``prompt_tokens_by_source_total{source=...}`` metric, dispatched
        # label-aware in ``_resolve_canonical`` (cache_hit vs local_compute).
        # cache_config_info is also NOT here — it's an info gauge whose value is
        # 1.0; the real ``num_gpu_blocks`` lives in a label and is extracted
        # separately in ``decode`` (label-as-value).
    }

    def __init__(self) -> None:
        self._store = self.store_cls.default()
        self._poll_count = 0
        # Previous cumulative snapshot per node — for windowed delta computation
        # in the periodic evidence log. {node_id: {canonical_key: value}}
        self._prev: dict[str, dict[str, float]] = {}

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
            # Label-aware parse: Prometheus exposition is
            #   ``metric_name{label="v",...}<ws>value``
            # We need labels for ``prompt_tokens_by_source_total{source=...}`` to
            # split cache-hit vs computed prompt tokens, and for
            # ``cache_config_info`` to read ``num_gpu_blocks``.
            try:
                if "{" in line:
                    name_part, rest = line.split("{", 1)
                    labels_str, _, value_part = rest.partition("}")
                    raw_name = name_part.strip()
                    labels: dict[str, str] = {}
                    for kv in labels_str.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            labels[k.strip()] = v.strip().strip('"')
                else:
                    # No-label line: ``name<ws>value`` — split once, reuse parts.
                    parts = line.split()
                    raw_name = parts[0]
                    labels = {}
                    value_part = parts[-1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                continue
            # cache_config_info is an info gauge: its value is 1.0, the real
            # data lives in the ``num_gpu_blocks`` label.
            if raw_name == "vllm:cache_config_info":
                n = labels.get("num_gpu_blocks")
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
            canonical = self._resolve_canonical(raw_name, labels)
            if canonical:
                value_type = METRIC_SPECS[canonical].get("value_type", float)
                result[canonical] = value_type(value)

        self._store.refresh({node_id: result})

        # Periodic visibility into what the collector fed the router — compare
        # against vllm's own "GPU KV cache usage" engine-stats log line.
        self._poll_count += 1
        logger.debug(f"vllm-metrics replica={node_id} polled: {result}")
        if self._poll_count % _METRICS_LOG_EVERY_POLLS == 0:
            self._log_evidence_window(node_id)

    @staticmethod
    def _resolve_canonical(raw_name: str, labels: dict[str, str]) -> str | None:
        """Map a scraped metric name (+labels) to a canonical key.

        Most metrics are label-free lookups in ``_PROMETHEUS_MAP``. The
        ``prompt_tokens_by_source_total`` metric carries a ``source`` label that
        splits prompt tokens into cache-hit vs locally-computed — the cleanest
        evidence signal for prefix reuse — so dispatch on the label.
        """
        if raw_name == "vllm:prompt_tokens_by_source_total":
            source = labels.get("source")
            if source == "local_cache_hit":
                return MetricKey.PROMPT_TOKENS_CACHED
            if source == "local_compute":
                return MetricKey.PROMPT_TOKENS
            return None
        return VLLMMetricsDecoder._PROMETHEUS_MAP.get(raw_name)

    def _log_evidence_window(self, node_id: str) -> None:
        """Emit a windowed evidence summary for one replica.

        Computes deltas vs the previous snapshot for cumulative counters/
        histograms so each line is a rate/average over ~``_METRICS_LOG_EVERY_POLLS``
        polls (≈30 s at the default 1 s interval). This is the raw feed for the
        B−A / D−C evidence chain (TTFT↓, prompt_tokens↓, cached↑ for kvcare).
        """
        # Read from the merged store snapshot (refresh already happened) rather
        # than the per-poll ``result`` — a transiently-missing scrape line would
        # otherwise zero a cumulative counter and corrupt the window delta.
        snap = self._store.get_metrics(node_id)
        prev = self._prev.get(node_id, {})

        def _delta(key: str) -> float:
            cur = float(snap.get(key, 0) or 0)
            return cur - float(prev.get(key, cur) or 0)

        kv = snap.get(MetricKey.KV_CACHE_USAGE_PERC)
        run = snap.get(MetricKey.NUM_REQUESTS_RUNNING)
        wait = snap.get(MetricKey.NUM_REQUESTS_WAITING)

        # Windowed TTFT/queue/TPOT averages (delta_sum / delta_count).
        ttft_avg = _avg(_delta(MetricKey.TTFT_SECONDS_SUM), _delta(MetricKey.TTFT_COUNT))
        queue_avg = _avg(_delta(MetricKey.QUEUE_TIME_SECONDS_SUM), _delta(MetricKey.QUEUE_TIME_COUNT))
        # prefill_time = TTFT - queue_wait. TTFT includes queue; subtracting it
        # isolates the real prefill compute cost that prefix-sharing reduces.
        prefill_t = (
            (ttft_avg - queue_avg)
            if (ttft_avg == ttft_avg and queue_avg == queue_avg)
            else float("nan")
        )
        tpot_avg = _avg(_delta(MetricKey.TPOT_SECONDS_SUM), _delta(MetricKey.TPOT_COUNT))

        # Token deltas over the window (prefill computed vs cached, decode, external).
        d_prefill = _delta(MetricKey.PROMPT_TOKENS)
        d_cached = _delta(MetricKey.PROMPT_TOKENS_CACHED)
        d_decode = _delta(MetricKey.GENERATION_TOKENS)
        d_external = _delta(MetricKey.EXTERNAL_PREFIX_CACHE_HITS)
        cache_hit_pct = (
            100.0 * d_cached / (d_cached + d_prefill) if (d_cached + d_prefill) > 0 else float("nan")
        )

        kv_str = f"{kv:.3f}" if isinstance(kv, float) else kv
        hit_str = f"{cache_hit_pct:.1f}" if cache_hit_pct == cache_hit_pct else "-"
        logger.info(
            f"vllm-evidence replica={node_id} kv={kv_str} run={run} wait={wait} | "
            f"TTFT={_ms(ttft_avg)}ms queue={_ms(queue_avg)}ms prefillT={_ms(prefill_t)}ms TPOT={_ms(tpot_avg)}ms | "
            f"prefill={int(d_prefill)} cached={int(d_cached)} (hit={hit_str}%) "
            f"decode={int(d_decode)} external={int(d_external)} [poll #{self._poll_count}]"
        )

        # Snapshot current cumulative values for next window's delta.
        self._prev[node_id] = {k: float(snap.get(k, 0) or 0) for k in _CUMULATIVE_KEYS}
