#!/usr/bin/env python3
"""Summarize paired baseline/SPEC-RL training benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIME_METRICS = {
    "trainer_step_s": "timing_s/step",
    "rollout_wait_s": "timing_s/gen",
    "actor_update_s": "timing_s/update_actor",
    "reward_s": "timing_s/reward",
    "old_logprob_s": "timing_s/old_log_prob",
    "weight_sync_s": "timing_s/update_weights",
}

SUM_METRICS = {
    "total_tokens": "perf/total_num_tokens",
    "prompt_tokens": "training/prompt_tokens",
    "response_tokens": "training/response_tokens",
    "draft_tokens": "spec/draft_tokens",
    "accepted_tokens": "spec/accepted_tokens",
    "verify_steps": "spec/verify_steps",
    "cache_hits": "spec/cache_hits",
    "cache_misses": "spec/cache_misses",
    "fallbacks": "spec/fallbacks",
    "saved_tokens": "spec/saved_tokens",
    "version_older_bypass": "spec/version_older_bypass",
    "version_equal_bypass": "spec/version_equal_bypass",
    "version_newer_verify": "spec/version_newer_verify",
    "verify_prompt_tokens": "spec/verify_prompt_tokens",
    "verify_draft_tokens": "spec/verify_draft_tokens",
    "continuation_tokens": "spec/continuation_tokens",
    "continuation_cached_tokens": "spec/continuation_cached_tokens",
    "continuation_prefill_tokens": "spec/continuation_prefill_tokens",
    "cache_lookup_ms": "spec/cache_lookup_ms",
    "version_check_ms": "spec/version_check_ms",
    "verify_ms": "spec/verify_ms",
    "continuation_ms": "spec/continuation_ms",
    "normal_fallback_ms": "spec/normal_fallback_ms",
    "fallback_unsupported": "spec/fallback/unsupported",
    "fallback_cache_miss": "spec/fallback/cache_miss",
    "fallback_older_policy": "spec/fallback/older_policy",
    "fallback_equal_policy": "spec/fallback/equal_policy",
    "fallback_verification_error": "spec/fallback/verification_error",
    "fallback_missing_logprobs": "spec/fallback/missing_logprobs",
    "fallback_continuation_error": "spec/fallback/continuation_error",
}

MEAN_METRICS = {
    "response_length_mean": "response_length/mean",
    "score_mean": "critic/score/mean",
    "reported_accept_rate": "rollout/spec_accept_rate",
    "reported_accept_length": "rollout/spec_accept_length",
}


@dataclass
class Trial:
    mode: str
    repeat: int
    directory: Path
    wall_seconds: float
    records: list[dict[str, Any]]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_trial(directory: Path) -> Trial:
    wall_path = directory / "wall.json"
    metrics_path = directory / "metrics.jsonl"
    if not wall_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"incomplete trial directory: {directory}")

    wall = json.loads(wall_path.read_text(encoding="utf-8"))
    if int(wall.get("exit_code", 1)) != 0:
        raise ValueError(f"trial failed with exit_code={wall.get('exit_code')}: {directory}")

    records = []
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {metrics_path}:{line_number}: {exc}") from exc
        data = payload.get("data")
        if isinstance(data, dict) and "timing_s/step" in data:
            records.append({"step": int(payload.get("step", 0)), "data": data})

    if not records:
        raise ValueError(f"no training-step metrics found in {metrics_path}")
    return Trial(
        mode=str(wall["mode"]),
        repeat=int(wall["repeat"]),
        directory=directory,
        wall_seconds=float(wall["wall_seconds"]),
        records=records,
    )


def _sum(records: list[dict[str, Any]], key: str) -> float:
    return sum(value for record in records if (value := _finite_float(record["data"].get(key))) is not None)


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [value for record in records if (value := _finite_float(record["data"].get(key))) is not None]
    return statistics.fmean(values) if values else None


def _summarize_trial(trial: Trial, warmup_steps: int) -> dict[str, Any]:
    ordered = sorted(trial.records, key=lambda record: record["step"])
    steady = [record for record in ordered if record["step"] > warmup_steps]
    if not steady:
        raise ValueError(
            f"trial {trial.directory} has no steps after warmup_steps={warmup_steps}; "
            f"available steps={[record['step'] for record in ordered]}"
        )

    result: dict[str, Any] = {
        "mode": trial.mode,
        "repeat": trial.repeat,
        "directory": str(trial.directory),
        "wall_seconds": trial.wall_seconds,
        "recorded_steps": len(ordered),
        "steady_steps": len(steady),
        "first_steady_step": steady[0]["step"],
        "last_steady_step": steady[-1]["step"],
    }
    for output_key, metric_key in TIME_METRICS.items():
        result[output_key] = _sum(steady, metric_key)
        result[f"all_{output_key}"] = _sum(ordered, metric_key)
    for output_key, metric_key in SUM_METRICS.items():
        result[output_key] = _sum(steady, metric_key)
    for output_key, metric_key in MEAN_METRICS.items():
        result[output_key] = _mean(steady, metric_key)

    lookups = result["cache_hits"] + result["cache_misses"]
    result["cache_lookups"] = lookups
    result["cache_hit_rate"] = result["cache_hits"] / lookups if lookups else 0.0
    result["accept_rate"] = result["accepted_tokens"] / result["draft_tokens"] if result["draft_tokens"] else 0.0
    result["saved_token_ratio"] = (
        result["saved_tokens"] / result["response_tokens"] if result["response_tokens"] else 0.0
    )
    result["verified_hits"] = result["version_newer_verify"]
    result["policy_version_bypasses"] = result["version_older_bypass"] + result["version_equal_bypass"]
    result["exception_fallbacks"] = (
        result["fallback_verification_error"]
        + result["fallback_missing_logprobs"]
        + result["fallback_continuation_error"]
    )
    result["verified_hit_rate"] = result["verified_hits"] / result["cache_hits"] if result["cache_hits"] else 0.0
    result["continuation_kv_reuse_rate"] = (
        result["continuation_cached_tokens"]
        / (result["continuation_cached_tokens"] + result["continuation_prefill_tokens"])
        if result["continuation_cached_tokens"] + result["continuation_prefill_tokens"]
        else 0.0
    )
    return result


def _aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for trial in trials
            for key, value in trial.items()
            if isinstance(value, (int, float)) and key not in {"repeat"}
        }
    )
    result: dict[str, Any] = {"trials": len(trials)}
    for key in numeric_keys:
        values = [float(trial[key]) for trial in trials if trial.get(key) is not None]
        if not values:
            continue
        result[key] = statistics.fmean(values)
        result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def _comparison(baseline: dict[str, Any], specrl: dict[str, Any], key: str) -> dict[str, float | None]:
    base = _finite_float(baseline.get(key))
    spec = _finite_float(specrl.get(key))
    if base is None or spec is None:
        return {"baseline": base, "specrl": spec, "reduction_seconds": None, "reduction_pct": None, "speedup": None}
    return {
        "baseline": base,
        "specrl": spec,
        "reduction_seconds": base - spec,
        "reduction_pct": ((base - spec) / base * 100.0) if base else None,
        "speedup": (base / spec) if spec else None,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _render_report(summary: dict[str, Any]) -> str:
    baseline = summary["aggregate"]["baseline"]
    specrl = summary["aggregate"]["specrl"]
    comparisons = summary["comparisons"]
    lines = [
        "# SPEC-RL Benchmark Report",
        "",
        f"- Repeats: {baseline['trials']}",
        f"- Warmup steps excluded from steady-state metrics: {summary['warmup_steps']}",
        "- End-to-end wall time includes model/Ray startup and all configured training work.",
        "- Trainer timing rows below aggregate only post-warmup training steps.",
        "",
        "## Time comparison",
        "",
        "| Metric | Baseline (s) | SPEC-RL (s) | Reduced (s) | Reduced (%) | Speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "wall_seconds": "End-to-end wall",
        "trainer_step_s": "Trainer steps",
        "rollout_wait_s": "Rollout generation/wait",
        "actor_update_s": "Actor update",
        "reward_s": "Reward",
        "old_logprob_s": "Old logprob",
        "weight_sync_s": "Weight sync",
    }
    for key, label in labels.items():
        item = comparisons[key]
        lines.append(
            f"| {label} | {_fmt(item['baseline'])} | {_fmt(item['specrl'])} | "
            f"{_fmt(item['reduction_seconds'])} | {_fmt(item['reduction_pct'], 2)} | {_fmt(item['speedup'])}x |"
        )

    lines.extend(
        [
            "",
            "## SPEC-RL diagnostics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Cache lookups | {_fmt(specrl.get('cache_lookups'), 0)} |",
            f"| Cache hits | {_fmt(specrl.get('cache_hits'), 0)} |",
            f"| Cache misses | {_fmt(specrl.get('cache_misses'), 0)} |",
            f"| Cache hit rate | {_fmt(100 * specrl.get('cache_hit_rate', 0), 2)}% |",
            f"| Verified cache hits | {_fmt(specrl.get('verified_hits'), 0)} |",
            f"| Verified / cache hits | {_fmt(100 * specrl.get('verified_hit_rate', 0), 2)}% |",
            f"| Older-policy bypasses | {_fmt(specrl.get('version_older_bypass'), 0)} |",
            f"| Equal-policy bypasses | {_fmt(specrl.get('version_equal_bypass'), 0)} |",
            f"| Policy-version bypasses | {_fmt(specrl.get('policy_version_bypasses'), 0)} |",
            f"| Exception fallbacks | {_fmt(specrl.get('exception_fallbacks'), 0)} |",
            f"| Fallbacks | {_fmt(specrl.get('fallbacks'), 0)} |",
            f"| Draft tokens | {_fmt(specrl.get('draft_tokens'), 0)} |",
            f"| Accepted tokens | {_fmt(specrl.get('accepted_tokens'), 0)} |",
            f"| Token accept rate | {_fmt(100 * specrl.get('accept_rate', 0), 2)}% |",
            f"| Framework-reported accept rate | {_fmt(100 * specrl.get('reported_accept_rate', 0), 2)}% |",
            f"| Verify steps | {_fmt(specrl.get('verify_steps'), 0)} |",
            f"| Saved tokens | {_fmt(specrl.get('saved_tokens'), 0)} |",
            f"| Saved / response tokens | {_fmt(100 * specrl.get('saved_token_ratio', 0), 2)}% |",
            f"| Reported accept length | {_fmt(specrl.get('reported_accept_length'))} |",
            "",
            "## SPEC-RL stage timing",
            "",
            "| Stage | Total (ms) |",
            "|---|---:|",
            f"| Cache lookup | {_fmt(specrl.get('cache_lookup_ms'))} |",
            f"| Version check | {_fmt(specrl.get('version_check_ms'))} |",
            f"| Draft verification | {_fmt(specrl.get('verify_ms'))} |",
            f"| Continuation | {_fmt(specrl.get('continuation_ms'))} |",
            f"| Normal fallback generation | {_fmt(specrl.get('normal_fallback_ms'))} |",
            "",
            "## SPEC-RL token work",
            "",
            "| Metric | Tokens |",
            "|---|---:|",
            f"| Verification prompt | {_fmt(specrl.get('verify_prompt_tokens'), 0)} |",
            f"| Verification draft | {_fmt(specrl.get('verify_draft_tokens'), 0)} |",
            f"| Continuation output | {_fmt(specrl.get('continuation_tokens'), 0)} |",
            f"| Continuation cached prefix | {_fmt(specrl.get('continuation_cached_tokens'), 0)} |",
            f"| Continuation re-prefill | {_fmt(specrl.get('continuation_prefill_tokens'), 0)} |",
            f"| Continuation KV reuse rate | {_fmt(100 * specrl.get('continuation_kv_reuse_rate', 0), 2)}% |",
            "",
            "## SPEC-RL route reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
            f"| unsupported | {_fmt(specrl.get('fallback_unsupported'), 0)} |",
            f"| cache_miss | {_fmt(specrl.get('fallback_cache_miss'), 0)} |",
            f"| older_policy | {_fmt(specrl.get('fallback_older_policy'), 0)} |",
            f"| equal_policy | {_fmt(specrl.get('fallback_equal_policy'), 0)} |",
            f"| verification_error | {_fmt(specrl.get('fallback_verification_error'), 0)} |",
            f"| missing_logprobs | {_fmt(specrl.get('fallback_missing_logprobs'), 0)} |",
            f"| continuation_error | {_fmt(specrl.get('fallback_continuation_error'), 0)} |",
            "",
            "## Token and outcome controls",
            "",
            "| Metric | Baseline | SPEC-RL |",
            "|---|---:|---:|",
            (
                f"| Total processed tokens | {_fmt(baseline.get('total_tokens'), 0)} | "
                f"{_fmt(specrl.get('total_tokens'), 0)} |"
            ),
            f"| Prompt tokens | {_fmt(baseline.get('prompt_tokens'), 0)} | {_fmt(specrl.get('prompt_tokens'), 0)} |",
            (
                f"| Response tokens | {_fmt(baseline.get('response_tokens'), 0)} | "
                f"{_fmt(specrl.get('response_tokens'), 0)} |"
            ),
            (
                f"| Mean response length | {_fmt(baseline.get('response_length_mean'))} | "
                f"{_fmt(specrl.get('response_length_mean'))} |"
            ),
            f"| Mean reward score | {_fmt(baseline.get('score_mean'))} | {_fmt(specrl.get('score_mean'))} |",
            "",
            "A negative reduction means SPEC-RL was slower for that metric.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()

    trial_dirs = sorted(
        path for path in args.benchmark_dir.iterdir() if path.is_dir() and (path / "wall.json").is_file()
    )
    trials = [_summarize_trial(_load_trial(path), args.warmup_steps) for path in trial_dirs]
    grouped = {mode: [trial for trial in trials if trial["mode"] == mode] for mode in ("baseline", "specrl")}
    if not grouped["baseline"] or not grouped["specrl"]:
        raise SystemExit("benchmark directory must contain successful baseline and specrl trials")
    if len(grouped["baseline"]) != len(grouped["specrl"]):
        raise SystemExit("baseline and specrl trial counts differ")

    aggregate = {mode: _aggregate(mode_trials) for mode, mode_trials in grouped.items()}
    comparison_keys = ["wall_seconds", *TIME_METRICS.keys()]
    summary = {
        "warmup_steps": args.warmup_steps,
        "trials": trials,
        "aggregate": aggregate,
        "comparisons": {key: _comparison(aggregate["baseline"], aggregate["specrl"], key) for key in comparison_keys},
    }
    summary_path = args.benchmark_dir / "summary.json"
    report_path = args.benchmark_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(summary)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Summary JSON: {summary_path}")
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
