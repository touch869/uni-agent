#!/usr/bin/env python3
"""Extract analysis-ready summaries from SPEC-RL benchmark logs.

The input can be a benchmark directory or a ZIP archive containing one. The
tool reads ``wall.json``, ``metrics.jsonl`` and ``console.log`` from every
``repeat_NN_{baseline,specrl}`` directory and writes:

* ``experiment_summary.json``: complete structured result;
* ``trials.csv``: one row per baseline/SPEC-RL trial;
* ``pairs.csv``: one row per paired repeat;
* ``experiment_summary.md``: a compact human-readable report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import stat
import statistics
import sys
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

# Direct execution adds only the script directory to sys.path. Add the
# repository root so the shared benchmark analyzer can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.blackbox_recipes.mini_swe_agent.analyze_specrl_benchmark import (
    MEAN_METRICS,
    SUM_METRICS,
    TIME_METRICS,
)

SCHEMA_VERSION = 1
TRIAL_RE = re.compile(r"^repeat_(\d+)_(baseline|specrl)$")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

CONFIG_PATTERNS = {
    "model": re.compile(r"^Model:\s+(.+)$", re.MULTILINE),
    "train_data": re.compile(r"^Train data:\s+(.+)$", re.MULTILINE),
    "val_data": re.compile(r"^Val data:\s+(.+)$", re.MULTILINE),
    "engine": re.compile(r"^Engine:\s+(.+)$", re.MULTILINE),
    "runner": re.compile(r"^Runner:\s+(.+)$", re.MULTILINE),
    "turns": re.compile(r"^Turns:\s+(.+)$", re.MULTILINE),
    "batch": re.compile(r"^Batch:\s+(.+)$", re.MULTILINE),
    "sequence": re.compile(r"^Sequence:\s+(.+)$", re.MULTILINE),
    "specrl": re.compile(r"^SPEC-RL:\s+(.+)$", re.MULTILINE),
    "trainer": re.compile(r"^Trainer:\s+(.+)$", re.MULTILINE),
    "resources": re.compile(r"^Resources:\s+(.+)$", re.MULTILINE),
    "samples": re.compile(r"^Samples:\s+(.+)$", re.MULTILINE),
}

ANOMALY_PATTERNS = {
    "cuda_oom": re.compile(r"OutOfMemoryError|CUDA out of memory|CUDA Error: out of memory", re.IGNORECASE),
    "actor_died": re.compile(r"ActorDiedError|worker died|Worker exits unexpectedly", re.IGNORECASE),
    "gloo_port_conflict": re.compile(r"EADDRINUSE|address already in use", re.IGNORECASE),
    "engine_start_failure": re.compile(r"EngineCore failed to start", re.IGNORECASE),
    "runner_failure": re.compile(r"Mini-swe-agent runner failed", re.IGNORECASE),
    "agent_parse_failure": re.compile(r"Failed to parse agent result", re.IGNORECASE),
    "post_setup_failure": re.compile(r"post_setup_cmd failed", re.IGNORECASE),
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
}

TRIAL_CSV_FIELDS = [
    "repeat",
    "mode",
    "status",
    "exit_code",
    "wall_seconds",
    "recorded_steps",
    "steady_steps",
    "trainer_step_s",
    "rollout_wait_s",
    "actor_update_s",
    "weight_sync_s",
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "total_tokens_per_s",
    "response_tokens_per_rollout_s",
    "score_mean",
    "reward_abs_max",
    "actor_loss_abs_max",
    "pg_loss_abs_max",
    "grad_norm_max",
    "response_clip_ratio_mean",
    "cache_hits",
    "cache_misses",
    "verified_hits",
    "version_equal_bypass",
    "version_older_bypass",
    "draft_tokens",
    "accepted_tokens",
    "saved_tokens",
    "accept_rate",
    "saved_token_ratio",
    "verify_ms",
    "continuation_ms",
    "continuation_tokens",
    "continuation_cached_tokens",
    "continuation_prefill_tokens",
    "exception_fallbacks",
    "agent_runs",
    "agent_limits_exceeded",
    "agent_nonempty_submissions",
    "eval_resolved_true",
    "eval_resolved_false",
    "cuda_oom",
    "actor_died",
    "gloo_port_conflict",
    "traceback",
    "directory",
]

PAIR_CSV_FIELDS = [
    "repeat",
    "baseline_status",
    "specrl_status",
    "wall_baseline_s",
    "wall_specrl_s",
    "wall_speedup",
    "trainer_baseline_s",
    "trainer_specrl_s",
    "trainer_speedup",
    "rollout_baseline_s",
    "rollout_specrl_s",
    "rollout_speedup",
    "baseline_total_tokens",
    "specrl_total_tokens",
    "tokens_equal",
    "specrl_cache_hits",
    "specrl_verified_hits",
    "specrl_equal_bypass",
    "specrl_accept_rate",
    "specrl_saved_token_ratio",
    "specrl_continuation_tokens",
    "specrl_exception_fallbacks",
    "baseline_reward_abs_max",
    "specrl_reward_abs_max",
    "baseline_grad_norm_max",
    "specrl_grad_norm_max",
]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sum(records: list[dict[str, Any]], key: str) -> float:
    return sum(value for record in records if (value := _finite_float(record["data"].get(key))) is not None)


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [value for record in records if (value := _finite_float(record["data"].get(key))) is not None]
    return statistics.fmean(values) if values else None


def _abs_max(records: list[dict[str, Any]], keys: tuple[str, ...]) -> float:
    values = [
        abs(value)
        for record in records
        for key in keys
        if (value := _finite_float(record["data"].get(key))) is not None
    ]
    return max(values, default=0.0)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _finite_float(numerator)
    bottom = _finite_float(denominator)
    return top / bottom if top is not None and bottom not in (None, 0.0) else None


def _parse_console(text: str) -> dict[str, Any]:
    clean = ANSI_RE.sub("", text)
    config = {
        key: match.group(1).strip()
        for key, pattern in CONFIG_PATTERNS.items()
        if (match := pattern.search(clean)) is not None
    }

    statuses: Counter[str] = Counter()
    submission_lengths: list[int] = []
    for match in re.finditer(r"agent: exit_status=([^,]+), submission=(\d+) chars", clean):
        statuses[match.group(1).strip()] += 1
        submission_lengths.append(int(match.group(2)))

    resolved: Counter[str] = Counter()
    for line in clean.splitlines():
        if "Eval report:" not in line:
            continue
        match = re.search(r"['\"]resolved['\"]:\s*(True|False|true|false)", line)
        if match:
            resolved[match.group(1).lower()] += 1

    anomaly_counts = {
        name: sum(1 for line in clean.splitlines() if pattern.search(line))
        for name, pattern in ANOMALY_PATTERNS.items()
    }
    return {
        "config": config,
        "agent": {
            "runs": sum(statuses.values()),
            "status_counts": dict(sorted(statuses.items())),
            "limits_exceeded": statuses.get("LimitsExceeded", 0),
            "empty_submissions": sum(length == 0 for length in submission_lengths),
            "nonempty_submissions": sum(length > 0 for length in submission_lengths),
        },
        "evaluation": {
            "resolved_true": resolved.get("true", 0),
            "resolved_false": resolved.get("false", 0),
        },
        "anomalies": anomaly_counts,
    }


def _load_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], ["metrics.jsonl missing"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON at line {line_number}: {exc.msg}")
            continue
        data = payload.get("data")
        if isinstance(data, dict) and "timing_s/step" in data:
            records.append({"step": int(payload.get("step", 0)), "data": data})
    if not records:
        errors.append("no training-step metrics")
    return records, errors


def _parse_trial(directory: Path, warmup_steps: int) -> dict[str, Any]:
    name_match = TRIAL_RE.match(directory.name)
    if not name_match:
        raise ValueError(f"invalid trial directory name: {directory}")
    repeat_from_name, mode_from_name = int(name_match.group(1)), name_match.group(2)

    wall_errors: list[str] = []
    wall: dict[str, Any] = {}
    wall_path = directory / "wall.json"
    if wall_path.is_file():
        try:
            wall = json.loads(wall_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            wall_errors.append(f"invalid wall.json: {exc}")
    else:
        wall_errors.append("wall.json missing")

    records, metric_errors = _load_records(directory / "metrics.jsonl")
    ordered = sorted(records, key=lambda record: record["step"])
    steady = [record for record in ordered if record["step"] > warmup_steps]
    if ordered and not steady:
        metric_errors.append(f"no steps after warmup={warmup_steps}")

    console_path = directory / "console.log"
    console = _parse_console(
        console_path.read_text(encoding="utf-8", errors="replace") if console_path.is_file() else ""
    )
    if not console_path.is_file():
        wall_errors.append("console.log missing")

    exit_code = int(wall.get("exit_code", 1)) if wall else None
    errors = [*wall_errors, *metric_errors]
    status_name = "success" if exit_code == 0 and steady and not metric_errors else "failed" if exit_code else "incomplete"
    result: dict[str, Any] = {
        "mode": str(wall.get("mode", mode_from_name)),
        "repeat": int(wall.get("repeat", repeat_from_name)),
        "directory": str(directory),
        "status": status_name,
        "exit_code": exit_code,
        "wall_seconds": _finite_float(wall.get("wall_seconds")),
        "recorded_steps": len(ordered),
        "steady_steps": len(steady),
        "first_step": ordered[0]["step"] if ordered else None,
        "last_step": ordered[-1]["step"] if ordered else None,
        "first_steady_step": steady[0]["step"] if steady else None,
        "last_steady_step": steady[-1]["step"] if steady else None,
        "parse_errors": errors,
        "config": console["config"],
        "agent": console["agent"],
        "evaluation": console["evaluation"],
        "anomalies": console["anomalies"],
    }

    for output_key, metric_key in TIME_METRICS.items():
        result[output_key] = _sum(steady, metric_key)
        result[f"all_{output_key}"] = _sum(ordered, metric_key)
    for output_key, metric_key in SUM_METRICS.items():
        result[output_key] = _sum(steady, metric_key)
    for output_key, metric_key in MEAN_METRICS.items():
        result[output_key] = _mean(steady, metric_key)

    result.update(
        {
            "reward_abs_max": _abs_max(steady, ("critic/rewards/min", "critic/rewards/max")),
            "score_abs_max": _abs_max(steady, ("critic/score/min", "critic/score/max")),
            "actor_loss_abs_max": _abs_max(steady, ("actor/loss",)),
            "pg_loss_abs_max": _abs_max(steady, ("actor/pg_loss",)),
            "grad_norm_max": _abs_max(steady, ("actor/grad_norm",)),
            "response_clip_ratio_mean": _mean(steady, "response_length/clip_ratio"),
            "aborted_ratio_mean": _mean(steady, "response/aborted_ratio"),
        }
    )
    lookups = result["cache_hits"] + result["cache_misses"]
    result["cache_lookups"] = lookups
    result["cache_hit_rate"] = _ratio(result["cache_hits"], lookups) or 0.0
    result["accept_rate"] = _ratio(result["accepted_tokens"], result["draft_tokens"]) or 0.0
    result["saved_token_ratio"] = _ratio(result["saved_tokens"], result["response_tokens"]) or 0.0
    result["verified_hits"] = result["version_newer_verify"]
    result["policy_version_bypasses"] = result["version_older_bypass"] + result["version_equal_bypass"]
    result["exception_fallbacks"] = (
        result["fallback_verification_error"]
        + result["fallback_missing_logprobs"]
        + result["fallback_continuation_error"]
    )
    result["continuation_kv_reuse_rate"] = (
        _ratio(
            result["continuation_cached_tokens"],
            result["continuation_cached_tokens"] + result["continuation_prefill_tokens"],
        )
        or 0.0
    )
    result["total_tokens_per_s"] = _ratio(result["total_tokens"], result["trainer_step_s"])
    result["response_tokens_per_rollout_s"] = _ratio(result["response_tokens"], result["rollout_wait_s"])
    return result


def _pair_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(trial["repeat"], trial["mode"]): trial for trial in trials}
    pairs: list[dict[str, Any]] = []
    for repeat in sorted({trial["repeat"] for trial in trials}):
        baseline = indexed.get((repeat, "baseline"))
        specrl = indexed.get((repeat, "specrl"))
        if not baseline or not specrl:
            continue
        pair = {
            "repeat": repeat,
            "baseline_status": baseline["status"],
            "specrl_status": specrl["status"],
            "wall_baseline_s": baseline["wall_seconds"],
            "wall_specrl_s": specrl["wall_seconds"],
            "wall_speedup": _ratio(baseline["wall_seconds"], specrl["wall_seconds"]),
            "trainer_baseline_s": baseline["trainer_step_s"],
            "trainer_specrl_s": specrl["trainer_step_s"],
            "trainer_speedup": _ratio(baseline["trainer_step_s"], specrl["trainer_step_s"]),
            "rollout_baseline_s": baseline["rollout_wait_s"],
            "rollout_specrl_s": specrl["rollout_wait_s"],
            "rollout_speedup": _ratio(baseline["rollout_wait_s"], specrl["rollout_wait_s"]),
            "baseline_total_tokens": baseline["total_tokens"],
            "specrl_total_tokens": specrl["total_tokens"],
            "tokens_equal": baseline["total_tokens"] == specrl["total_tokens"],
            "specrl_cache_hits": specrl["cache_hits"],
            "specrl_verified_hits": specrl["verified_hits"],
            "specrl_equal_bypass": specrl["version_equal_bypass"],
            "specrl_accept_rate": specrl["accept_rate"],
            "specrl_saved_token_ratio": specrl["saved_token_ratio"],
            "specrl_continuation_tokens": specrl["continuation_tokens"],
            "specrl_exception_fallbacks": specrl["exception_fallbacks"],
            "baseline_reward_abs_max": baseline["reward_abs_max"],
            "specrl_reward_abs_max": specrl["reward_abs_max"],
            "baseline_grad_norm_max": baseline["grad_norm_max"],
            "specrl_grad_norm_max": specrl["grad_norm_max"],
        }
        pairs.append(pair)
    return pairs


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate(trials: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    valid_pairs = [
        pair
        for pair in pairs
        if pair["baseline_status"] == "success" and pair["specrl_status"] == "success"
    ]
    paired_metrics = {}
    for key in ("wall_speedup", "trainer_speedup", "rollout_speedup"):
        paired_metrics[key] = _stats(
            [value for pair in valid_pairs if (value := _finite_float(pair.get(key))) is not None]
        )
    return {
        "trial_count": len(trials),
        "successful_trials": sum(trial["status"] == "success" for trial in trials),
        "pair_count": len(pairs),
        "successful_pairs": len(valid_pairs),
        "all_pair_tokens_equal": bool(valid_pairs) and all(pair["tokens_equal"] for pair in valid_pairs),
        "paired_metrics": paired_metrics,
        "total_agent_runs": sum(trial["agent"]["runs"] for trial in trials),
        "total_limits_exceeded": sum(trial["agent"]["limits_exceeded"] for trial in trials),
        "total_resolved": sum(trial["evaluation"]["resolved_true"] for trial in trials),
        "total_unresolved": sum(trial["evaluation"]["resolved_false"] for trial in trials),
        "total_exception_fallbacks": sum(trial["exception_fallbacks"] for trial in trials),
        "anomalies": {
            key: sum(trial["anomalies"].get(key, 0) for trial in trials) for key in ANOMALY_PATTERNS
        },
    }


def _findings(trials: list[dict[str, Any]], pairs: list[dict[str, Any]], aggregate: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if aggregate["successful_trials"] != aggregate["trial_count"]:
        findings.append({"severity": "error", "code": "failed_trials", "message": "存在失败或不完整 trial。"})
    if aggregate["successful_pairs"] and not aggregate["all_pair_tokens_equal"]:
        findings.append({"severity": "error", "code": "token_mismatch", "message": "至少一个配对的 total tokens 不一致。"})
    if trials and all(
        trial["reward_abs_max"] == 0 and trial["actor_loss_abs_max"] == 0 and trial["grad_norm_max"] == 0
        for trial in trials
        if trial["status"] == "success"
    ):
        findings.append(
            {
                "severity": "warning",
                "code": "zero_training_signal",
                "message": "所有成功 trial 的 reward、actor loss 和 grad norm 均为 0；结果属于无有效权重变化上界。",
            }
        )
    if aggregate["total_agent_runs"] and aggregate["total_limits_exceeded"] == aggregate["total_agent_runs"]:
        findings.append(
            {
                "severity": "warning",
                "code": "all_agent_limits_exceeded",
                "message": "所有已记录 agent run 均以 LimitsExceeded 结束。",
            }
        )
    if any((trial["response_clip_ratio_mean"] or 0) >= 0.999 for trial in trials):
        findings.append(
            {
                "severity": "info",
                "code": "responses_fully_clipped",
                "message": "至少一个 trial 的 response clip ratio 为 100%。",
            }
        )
    if any(pair["specrl_continuation_tokens"] > 0 for pair in pairs):
        findings.append({"severity": "info", "code": "continuation_observed", "message": "实验实际触发了 continuation。"})
    if aggregate["total_exception_fallbacks"]:
        findings.append(
            {
                "severity": "error",
                "code": "exception_fallbacks",
                "message": f"检测到 {aggregate['total_exception_fallbacks']:.0f} 次 SPEC-RL 异常 fallback。",
            }
        )
    for name, count in aggregate["anomalies"].items():
        if count and name != "traceback":
            findings.append(
                {"severity": "error", "code": name, "message": f"console.log 中检测到 {count} 行 {name}。"}
            )
    return findings


def _infer_config(trials: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    configs = [trial["config"] for trial in trials if trial["config"]]
    if not configs:
        return {}, []
    canonical = configs[0]
    mismatches = sorted(
        key
        for key in set().union(*(config.keys() for config in configs))
        if key != "specrl" and len({config.get(key) for config in configs}) > 1
    )
    return canonical, mismatches


def analyze_experiment(benchmark_dir: Path, warmup_steps: int) -> dict[str, Any]:
    trial_dirs = sorted(
        path
        for path in benchmark_dir.iterdir()
        if path.is_dir() and TRIAL_RE.match(path.name) and any((path / name).exists() for name in ("wall.json", "metrics.jsonl"))
    )
    if not trial_dirs:
        raise ValueError(f"no repeat_NN_baseline/specrl directories found in {benchmark_dir}")
    trials = [_parse_trial(directory, warmup_steps) for directory in trial_dirs]
    pairs = _pair_trials(trials)
    aggregate = _aggregate(trials, pairs)
    config, config_mismatches = _infer_config(trials)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": benchmark_dir.name,
        "benchmark_directory": str(benchmark_dir),
        "warmup_steps": warmup_steps,
        "config": config,
        "config_mismatches": config_mismatches,
        "trials": trials,
        "pairs": pairs,
        "aggregate": aggregate,
        "findings": _findings(trials, pairs, aggregate),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _render_markdown(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        f"# SPEC-RL Experiment Extraction: {result['benchmark_name']}",
        "",
        f"- Warmup steps: {result['warmup_steps']}",
        f"- Trials: {aggregate['successful_trials']}/{aggregate['trial_count']} successful",
        f"- Pairs: {aggregate['successful_pairs']}/{aggregate['pair_count']} successful",
        f"- Paired token equality: {'yes' if aggregate['all_pair_tokens_equal'] else 'no'}",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in result["config"].items():
        lines.append(f"| {key} | {value} |")
    if result["config_mismatches"]:
        lines.append(f"\nConfig mismatches: {', '.join(result['config_mismatches'])}")

    lines.extend(
        [
            "",
            "## Paired Performance",
            "",
            "| Repeat | Status B/S | Wall B/S (s) | Wall x | Trainer B/S (s) | Trainer x | Rollout B/S (s) | Rollout x | Tokens equal |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for pair in result["pairs"]:
        lines.append(
            f"| {pair['repeat']} | {pair['baseline_status']}/{pair['specrl_status']} | "
            f"{_fmt(pair['wall_baseline_s'])}/{_fmt(pair['wall_specrl_s'])} | {_fmt(pair['wall_speedup'])}x | "
            f"{_fmt(pair['trainer_baseline_s'])}/{_fmt(pair['trainer_specrl_s'])} | {_fmt(pair['trainer_speedup'])}x | "
            f"{_fmt(pair['rollout_baseline_s'])}/{_fmt(pair['rollout_specrl_s'])} | {_fmt(pair['rollout_speedup'])}x | "
            f"{'yes' if pair['tokens_equal'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## SPEC-RL Per Repeat",
            "",
            "| Repeat | Cache hits | Verified | Equal bypass | Accept rate | Saved/response | Continuation tokens | Exception fallbacks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in result["pairs"]:
        lines.append(
            f"| {pair['repeat']} | {_fmt(pair['specrl_cache_hits'], 0)} | {_fmt(pair['specrl_verified_hits'], 0)} | "
            f"{_fmt(pair['specrl_equal_bypass'], 0)} | {_fmt(100 * pair['specrl_accept_rate'], 2)}% | "
            f"{_fmt(100 * pair['specrl_saved_token_ratio'], 2)}% | {_fmt(pair['specrl_continuation_tokens'], 0)} | "
            f"{_fmt(pair['specrl_exception_fallbacks'], 0)} |"
        )

    lines.extend(
        [
            "",
            "## Training and Agent Health",
            "",
            "| Trial | Reward max | Actor loss max | Grad norm max | Clip ratio | Agent runs | LimitsExceeded | Resolved T/F |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trial in result["trials"]:
        lines.append(
            f"| repeat {trial['repeat']} {trial['mode']} | {_fmt(trial['reward_abs_max'])} | "
            f"{_fmt(trial['actor_loss_abs_max'])} | {_fmt(trial['grad_norm_max'])} | "
            f"{_fmt(100 * (trial['response_clip_ratio_mean'] or 0), 2)}% | {trial['agent']['runs']} | "
            f"{trial['agent']['limits_exceeded']} | {trial['evaluation']['resolved_true']}/{trial['evaluation']['resolved_false']} |"
        )

    lines.extend(["", "## Findings", ""])
    if result["findings"]:
        lines.extend(
            f"- **{finding['severity'].upper()} [{finding['code']}]** {finding['message']}"
            for finding in result["findings"]
        )
    else:
        lines.append("- No automatic findings.")

    lines.extend(["", "## Console Anomalies", "", "| Type | Matching lines |", "|---|---:|"])
    for key, count in aggregate["anomalies"].items():
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines) + "\n"


def _flatten_trial(trial: dict[str, Any]) -> dict[str, Any]:
    row = {key: trial.get(key) for key in TRIAL_CSV_FIELDS}
    row.update(
        {
            "agent_runs": trial["agent"]["runs"],
            "agent_limits_exceeded": trial["agent"]["limits_exceeded"],
            "agent_nonempty_submissions": trial["agent"]["nonempty_submissions"],
            "eval_resolved_true": trial["evaluation"]["resolved_true"],
            "eval_resolved_false": trial["evaluation"]["resolved_false"],
        }
    )
    for key in ANOMALY_PATTERNS:
        row[key] = trial["anomalies"].get(key, 0)
    return row


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "experiment_summary.json",
        "trials_csv": output_dir / "trials.csv",
        "pairs_csv": output_dir / "pairs.csv",
        "markdown": output_dir / "experiment_summary.md",
    }
    paths["json"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["trials_csv"], TRIAL_CSV_FIELDS, [_flatten_trial(trial) for trial in result["trials"]])
    _write_csv(paths["pairs_csv"], PAIR_CSV_FIELDS, result["pairs"])
    paths["markdown"].write_text(_render_markdown(result), encoding="utf-8")
    return paths


def _discover_benchmark_root(root: Path) -> Path:
    trial_dirs = sorted(
        path for path in root.rglob("repeat_*_*") if path.is_dir() and TRIAL_RE.match(path.name)
    )
    if not trial_dirs:
        raise ValueError(f"archive/directory contains no trial directories: {root}")
    parents = {path.parent.resolve() for path in trial_dirs}
    if len(parents) != 1:
        raise ValueError(f"input contains multiple benchmark roots: {sorted(map(str, parents))}")
    return next(iter(parents))


def _safe_extract_zip(archive: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symlink is not supported: {info.filename}")
            destination = target.joinpath(*relative.parts).resolve()
            if not destination.is_relative_to(target):
                raise ValueError(f"unsafe ZIP destination: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)


@contextmanager
def materialize_input(input_path: Path) -> Iterator[Path]:
    if input_path.is_dir():
        yield _discover_benchmark_root(input_path)
        return
    if input_path.is_file() and zipfile.is_zipfile(input_path):
        with tempfile.TemporaryDirectory(prefix="specrl-log-extract-") as temp_dir:
            root = Path(temp_dir)
            _safe_extract_zip(input_path, root)
            yield _discover_benchmark_root(root)
        return
    raise ValueError(f"input must be a benchmark directory or ZIP archive: {input_path}")


def _infer_warmup_steps(benchmark_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    summary_path = benchmark_dir / "summary.json"
    if summary_path.is_file():
        try:
            value = int(json.loads(summary_path.read_text(encoding="utf-8"))["warmup_steps"])
            return value
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="benchmark directory or ZIP archive")
    parser.add_argument("--warmup-steps", type=int, default=None, help="override warmup steps")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output root; results are written to <output-dir>/<experiment-name>",
    )
    args = parser.parse_args()
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    input_path = args.input.resolve()
    default_output = (
        input_path / "log_analysis" if input_path.is_dir() else input_path.parent / f"{input_path.stem}_analysis"
    )
    output_root = (args.output_dir or default_output).resolve()
    with materialize_input(input_path) as benchmark_dir:
        warmup_steps = _infer_warmup_steps(benchmark_dir, args.warmup_steps)
        result = analyze_experiment(benchmark_dir, warmup_steps)
        result["source"] = str(input_path)
        result["benchmark_directory"] = str(input_path if input_path.is_dir() else benchmark_dir.name)
        if input_path.is_file():
            for trial in result["trials"]:
                trial["directory"] = f"{input_path}!/{benchmark_dir.name}/{Path(trial['directory']).name}"
        output_dir = output_root / result["benchmark_name"]
        paths = write_outputs(result, output_dir)

    print(f"Experiment: {result['benchmark_name']}")
    print(f"Successful pairs: {result['aggregate']['successful_pairs']}/{result['aggregate']['pair_count']}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
