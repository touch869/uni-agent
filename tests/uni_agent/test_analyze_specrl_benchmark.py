from __future__ import annotations

import json

from examples.blackbox_recipes.mini_swe_agent.analyze_specrl_benchmark import (
    _aggregate,
    _comparison,
    _load_trial,
    _render_report,
    _summarize_trial,
)


def _write_trial(tmp_path, mode, wall_seconds, step_seconds, spec_metrics=None):
    directory = tmp_path / f"repeat_01_{mode}"
    directory.mkdir()
    (directory / "wall.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "repeat": 1,
                "wall_seconds": wall_seconds,
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    records = []
    for step in range(1, 5):
        data = {
            "timing_s/step": step_seconds,
            "timing_s/gen": step_seconds * 0.6,
            "timing_s/update_actor": step_seconds * 0.2,
            "perf/total_num_tokens": 100,
            "training/prompt_tokens": 40,
            "training/response_tokens": 60,
            "response_length/mean": 30,
            "critic/score/mean": 0.5,
        }
        data.update(spec_metrics or {})
        records.append(json.dumps({"step": step, "data": data}))
    (directory / "metrics.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    return directory


def test_benchmark_analyzer_summarizes_time_cache_and_tokens(tmp_path):
    baseline_dir = _write_trial(tmp_path, "baseline", wall_seconds=100, step_seconds=10)
    specrl_dir = _write_trial(
        tmp_path,
        "specrl",
        wall_seconds=80,
        step_seconds=7,
        spec_metrics={
            "spec/cache_hits": 3,
            "spec/cache_misses": 1,
            "spec/fallbacks": 0,
            "spec/draft_tokens": 50,
            "spec/accepted_tokens": 40,
            "spec/verify_steps": 2,
            "spec/saved_tokens": 40,
            "spec/version_older_bypass": 1,
            "spec/version_equal_bypass": 0,
            "spec/version_newer_verify": 2,
            "spec/verify_ms": 12.5,
            "spec/continuation_ms": 7.5,
            "spec/continuation_cached_tokens": 90,
            "spec/continuation_prefill_tokens": 10,
            "spec/fallback/verification_error": 1,
            "rollout/spec_accept_rate": 0.8,
            "rollout/spec_accept_length": 21,
        },
    )

    baseline = _summarize_trial(_load_trial(baseline_dir), warmup_steps=1)
    specrl = _summarize_trial(_load_trial(specrl_dir), warmup_steps=1)
    assert baseline["steady_steps"] == 3
    assert baseline["trainer_step_s"] == 30
    assert baseline["version_newer_verify"] == 0
    assert baseline["verify_ms"] == 0
    assert specrl["trainer_step_s"] == 21
    assert specrl["cache_hits"] == 9
    assert specrl["cache_misses"] == 3
    assert specrl["cache_lookups"] == 12
    assert specrl["cache_hit_rate"] == 0.75
    assert specrl["accepted_tokens"] == 120
    assert specrl["draft_tokens"] == 150
    assert specrl["accept_rate"] == 0.8
    assert specrl["verified_hits"] == 6
    assert specrl["policy_version_bypasses"] == 3
    assert specrl["exception_fallbacks"] == 3
    assert specrl["verify_ms"] == 37.5
    assert specrl["continuation_ms"] == 22.5
    assert specrl["continuation_kv_reuse_rate"] == 0.9

    aggregate = {"baseline": _aggregate([baseline]), "specrl": _aggregate([specrl])}
    comparison = _comparison(aggregate["baseline"], aggregate["specrl"], "trainer_step_s")
    assert comparison["reduction_seconds"] == 9
    assert comparison["reduction_pct"] == 30

    summary = {
        "warmup_steps": 1,
        "aggregate": aggregate,
        "comparisons": {
            key: _comparison(aggregate["baseline"], aggregate["specrl"], key)
            for key in (
                "wall_seconds",
                "trainer_step_s",
                "rollout_wait_s",
                "actor_update_s",
                "reward_s",
                "old_logprob_s",
                "weight_sync_s",
            )
        },
    }
    report = _render_report(summary)
    assert "End-to-end wall" in report
    assert "Cache hit rate" in report
    assert "Framework-reported accept rate" in report
    assert "Verified cache hits" in report
    assert "SPEC-RL stage timing" in report
    assert "Continuation KV reuse rate" in report
    assert "verification_error" in report
    assert "75.00%" in report
