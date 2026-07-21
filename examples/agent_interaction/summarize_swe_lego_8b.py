#!/usr/bin/env python3
"""Summarize SWE-Lego result JSONs without inventing missing evidence."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(path).read_text()) for path in args.result]
    vanilla = rows[0]
    vanilla_ids = [x["instance_id"] for x in vanilla["instances"]]
    summary = []
    for row in rows:
        ids = [x["instance_id"] for x in row["instances"]]
        summary.append(
            {
                "run_name": row["run_name"],
                "num_rollouts": row["num_samples"],
                "unique_instances": row["unique_instances"],
                "generation_time_s": row["generation_time_s"],
                "wall_time_s": row["wall_time_s"],
                "generation_speedup_vs_swe_lego_vanilla": vanilla["generation_time_s"]
                / row["generation_time_s"],
                "wall_speedup_vs_swe_lego_vanilla": vanilla["wall_time_s"] / row["wall_time_s"],
                "resolved": row["resolved"],
                "mean_rm_score": row["mean_rm_score"],
                "instance_ids_match_vanilla": ids == vanilla_ids,
                "evidence_complete": row["evidence_complete"],
            }
        )
    cached = [x["generation_time_s"] for x in summary if "cached" in x["run_name"]]
    output = {"runs": summary}
    if len(cached) >= 2:
        output["cached_statistics"] = {
            "mean_generation_time_s": statistics.mean(cached),
            "sample_stdev_s": statistics.stdev(cached),
            "cv": statistics.stdev(cached) / statistics.mean(cached),
            "range_s": max(cached) - min(cached),
        }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
