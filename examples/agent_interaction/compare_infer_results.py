#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_result(path):
    with open(path) as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def fmt_float(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Compare Uni-Agent parallel_infer result JSON files.")
    parser.add_argument("results", nargs="+", help="Result JSON files. First file is treated as baseline.")
    args = parser.parse_args()

    results = [load_result(Path(p)) for p in args.results]
    baseline = results[0]
    baseline_wall = baseline.get("wall_time_s")
    baseline_gen = baseline.get("generation_time_s")
    baseline_score = baseline.get("mean_rm_score", 0.0)

    print("run	n	mean_rm	resolved	wall_s	gen_s	wall_speedup	gen_speedup	path")
    for result in results:
        scores = result.get("rm_scores", [])
        resolved = sum(1 for score in scores if score > 0)
        wall = result.get("wall_time_s")
        gen = result.get("generation_time_s")
        wall_speedup = baseline_wall / wall if baseline_wall and wall else None
        gen_speedup = baseline_gen / gen if baseline_gen and gen else None
        run_name = Path(result["_path"]).stem
        print(
            "	".join(
                [
                    run_name,
                    str(result.get("num_samples", len(scores))),
                    fmt_float(result.get("mean_rm_score", 0.0)),
                    str(resolved),
                    fmt_float(wall, 2),
                    fmt_float(gen, 2),
                    fmt_float(wall_speedup, 3),
                    fmt_float(gen_speedup, 3),
                    result["_path"],
                ]
            )
        )

    if len(results) <= 1:
        return

    print("\nDelta vs baseline:")
    for result in results[1:]:
        name = Path(result["_path"]).stem
        score_delta = result.get("mean_rm_score", 0.0) - baseline_score
        wall = result.get("wall_time_s")
        gen = result.get("generation_time_s")
        wall_delta = (wall - baseline_wall) if wall is not None and baseline_wall is not None else None
        gen_delta = (gen - baseline_gen) if gen is not None and baseline_gen is not None else None
        print(
            f"{name}: mean_rm_delta={score_delta:+.4f}, "
            f"wall_delta_s={fmt_float(wall_delta, 2)}, gen_delta_s={fmt_float(gen_delta, 2)}"
        )


if __name__ == "__main__":
    main()
