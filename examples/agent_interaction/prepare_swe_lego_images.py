#!/usr/bin/env python3
"""Pre-pull the public mirror of SWE-Bench images needed by one phase."""

import argparse
import json
import subprocess
from pathlib import Path

from datasets import load_dataset


def public_image(image: str) -> str:
    private = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/"
    public = "enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/"
    return public + image[len(private) :] if image.startswith(private) else image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    samples = load_dataset("parquet", data_files=args.data_path, split="train").to_list()[: args.max_samples]
    images = []
    for sample in samples:
        image = sample["extra_info"]["tools_kwargs"]["env"]["deployment"]["image"]
        image = public_image(image)
        if image not in images:
            images.append(image)
    records = []
    for image in images:
        inspect = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
        if inspect.returncode == 0:
            records.append({"image": image, "status": "already_present"})
            continue
        pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        records.append(
            {
                "image": image,
                "status": "pulled" if pull.returncode == 0 else "pull_failed",
                "exit_code": pull.returncode,
                "stdout": pull.stdout,
                "stderr": pull.stderr,
            }
        )
        if pull.returncode != 0:
            Path(args.manifest).write_text(json.dumps(records, indent=2) + "\n")
            raise SystemExit(f"Failed to pull {image}: {pull.stderr}")
    Path(args.manifest).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
