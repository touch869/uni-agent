from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from examples.blackbox_recipes.mini_swe_agent.extract_specrl_logs import (
    analyze_experiment,
    materialize_input,
    write_outputs,
)


def _write_trial(root, mode, repeat, wall_seconds, step_seconds, *, specrl=False):
    directory = root / f"repeat_{repeat:02d}_{mode}"
    directory.mkdir(parents=True)
    (directory / "wall.json").write_text(
        json.dumps({"mode": mode, "repeat": repeat, "wall_seconds": wall_seconds, "exit_code": 0}),
        encoding="utf-8",
    )
    rows = []
    for step in range(1, 4):
        data = {
            "timing_s/step": step_seconds,
            "timing_s/gen": step_seconds * 0.6,
            "timing_s/update_actor": step_seconds * 0.2,
            "timing_s/update_weights": step_seconds * 0.1,
            "perf/total_num_tokens": 100,
            "training/prompt_tokens": 60,
            "training/response_tokens": 40,
            "response_length/mean": 20,
            "response_length/clip_ratio": 1.0,
            "critic/score/mean": 0.0,
            "critic/score/min": 0.0,
            "critic/score/max": 0.0,
            "critic/rewards/min": 0.0,
            "critic/rewards/max": 0.0,
            "actor/loss": 0.0,
            "actor/pg_loss": 0.0,
            "actor/grad_norm": 0.0,
        }
        if specrl:
            data.update(
                {
                    "spec/cache_hits": 2,
                    "spec/cache_misses": 0,
                    "spec/version_newer_verify": 1,
                    "spec/version_equal_bypass": 1,
                    "spec/draft_tokens": 40,
                    "spec/accepted_tokens": 30,
                    "spec/saved_tokens": 30,
                    "spec/continuation_tokens": 10,
                    "spec/continuation_cached_tokens": 90,
                    "spec/continuation_prefill_tokens": 10,
                    "spec/verify_ms": 5,
                }
            )
        rows.append(json.dumps({"step": step, "data": data}))
    (directory / "metrics.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (directory / "console.log").write_text(
        "\n".join(
            [
                "Model:       /models/test",
                "Engine:      vllm (gen_tp=2, train_tp=2)",
                "Turns:       agent_max_turns=1",
                "Batch:       n=2, mini_bsz=1",
                "Sequence:    prompt=1024, response=40",
                "[sample 0] agent: exit_status=LimitsExceeded, submission=0 chars",
                "Eval report: {'resolved': False}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_extracts_pairs_training_health_and_outputs(tmp_path):
    benchmark = tmp_path / "experiment"
    benchmark.mkdir()
    _write_trial(benchmark, "baseline", 1, 100, 10)
    _write_trial(benchmark, "specrl", 1, 80, 8, specrl=True)

    result = analyze_experiment(benchmark, warmup_steps=1)
    pair = result["pairs"][0]
    assert pair["wall_speedup"] == 1.25
    assert pair["trainer_speedup"] == 1.25
    assert pair["tokens_equal"] is True
    assert pair["specrl_cache_hits"] == 4
    assert pair["specrl_verified_hits"] == 2
    assert pair["specrl_continuation_tokens"] == 20
    assert result["aggregate"]["total_limits_exceeded"] == 2
    assert {finding["code"] for finding in result["findings"]} >= {
        "zero_training_signal",
        "all_agent_limits_exceeded",
        "responses_fully_clipped",
        "continuation_observed",
    }

    paths = write_outputs(result, tmp_path / "analysis")
    assert all(path.is_file() for path in paths.values())
    assert "Paired Performance" in paths["markdown"].read_text(encoding="utf-8")
    assert "wall_speedup" in paths["pairs_csv"].read_text(encoding="utf-8")


def test_reads_benchmark_from_zip(tmp_path):
    benchmark = tmp_path / "packed_experiment"
    benchmark.mkdir()
    _write_trial(benchmark, "baseline", 1, 100, 10)
    _write_trial(benchmark, "specrl", 1, 90, 9, specrl=True)
    archive = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in benchmark.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(tmp_path))

    with materialize_input(archive) as extracted:
        result = analyze_experiment(extracted, warmup_steps=1)
    assert result["benchmark_name"] == "packed_experiment"
    assert result["aggregate"]["successful_pairs"] == 1


def test_direct_script_execution_from_other_directory(tmp_path):
    benchmark = tmp_path / "direct_experiment"
    benchmark.mkdir()
    _write_trial(benchmark, "baseline", 1, 100, 10)
    _write_trial(benchmark, "specrl", 1, 90, 9, specrl=True)
    output_root = tmp_path / "analysis"
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/blackbox_recipes/mini_swe_agent/extract_specrl_logs.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(benchmark), "--output-dir", str(output_root)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result_dir = output_root / benchmark.name
    assert "json:" in completed.stdout
    assert (result_dir / "experiment_summary.json").is_file()
