#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-1.7B}"
export DATA_PATH="${DATA_PATH:-/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet}"

export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-${OPENYUANRONG_SERVER_ADDRESS:-}}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-${OPENYUANRONG_TOKEN:-}}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

python - <<'PY'
import importlib.metadata as md
import os
from pathlib import Path

import akernel_sdk
import pyarrow.parquet as pq
import ray
import torch
import vllm

model_path = Path(os.environ["MODEL_PATH"]).expanduser()
data_path = Path(os.environ["DATA_PATH"]).expanduser()
sandbox_api = Path(akernel_sdk.__file__).with_name("sandbox_api.py")
source = sandbox_api.read_text()

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
print(f"ray={ray.__version__}")
print(f"vllm={vllm.__version__}")
print(f"akernel-sdk={md.version('akernel-sdk')} path={sandbox_api}")
print(f"akernel_tunnel_wss={'wss://' in source}")
print(f"model_config={model_path / 'config.json'} exists={(model_path / 'config.json').exists()}")
print(f"data={data_path} exists={data_path.exists()}")
if data_path.exists():
    table = pq.read_table(data_path, columns=["prompt", "extra_info"])
    print(f"data_rows={table.num_rows}")
print(f"akernel_server_set={bool(os.environ.get('AKERNEL_SERVER_ADDRESS'))}")
print(f"akernel_token_set={bool(os.environ.get('AKERNEL_TOKEN'))}")

missing = []
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    missing.append("CUDA GPU")
if "wss://" not in source:
    missing.append("akernel_sdk wss tunnel patch")
if not (model_path / "config.json").exists():
    missing.append("model config")
if not data_path.exists():
    missing.append("dataset parquet")
if not os.environ.get("AKERNEL_SERVER_ADDRESS"):
    missing.append("AKERNEL_SERVER_ADDRESS or OPENYUANRONG_SERVER_ADDRESS")
if not os.environ.get("AKERNEL_TOKEN"):
    missing.append("AKERNEL_TOKEN or OPENYUANRONG_TOKEN")

if missing:
    print("missing=" + ", ".join(missing))
    raise SystemExit(2)

print("mini_swe_env_check=OK")
PY
