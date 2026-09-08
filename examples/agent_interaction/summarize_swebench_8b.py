#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="/workspace/output/swebench_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    runs = [
        ("vanilla", out_dir / f"vanilla_{args.tag}.json"),
        ("specrl_warm", out_dir / f"specrl_warm_{args.tag}.json"),
        ("specrl_cached_round1", out_dir / f"specrl_cached_{args.tag}_round1.json"),
        ("specrl_cached_round2", out_dir / f"specrl_cached_{args.tag}_round2.json"),
        ("specrl_cached_round3", out_dir / f"specrl_cached_{args.tag}_round3.json"),
    ]
    available = [(name, path) for name, path in runs if path.exists()]
    if not available or available[0][0] != "vanilla":
        raise SystemExit(f"vanilla result missing for tag {args.tag}")

    loaded = [(name, json.loads(path.read_text())) for name, path in available]
    vanilla = loaded[0][1]
    base_gen = vanilla["generation_time_s"]
    base_wall = vanilla["wall_time_s"]
    base_ids = [sample.get("instance_id") for sample in vanilla.get("samples", [])]
    rows = []
    for name, result in loaded:
        ids = [sample.get("instance_id") for sample in result.get("samples", [])]
        scores = result.get("rm_scores", [])
        generation = result.get("generation_time_s")
        wall = result.get("wall_time_s")
        rows.append(
            {
                "run": name,
                "num_samples": result.get("num_samples"),
                "unique_instances": len(set(ids)),
                "n": result.get("n"),
                "max_turns": result.get("max_turns"),
                "generation_time_s": generation,
                "wall_time_s": wall,
                "generation_speedup_vs_8b_vanilla": base_gen / generation if generation else None,
                "wall_speedup_vs_8b_vanilla": base_wall / wall if wall else None,
                "resolved": sum(score > 0 for score in scores),
                "mean_rm_score": result.get("mean_rm_score"),
                "instance_ids_match_8b_vanilla": ids == base_ids,
            }
        )

    summary_path = out_dir / f"summary_{args.tag}.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
