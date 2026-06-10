#!/bin/bash
# Run inference. Can be executed from anywhere:

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

MODEL_PATH=${1:-/path/to/Qwen3-4B}
DATA_PATH=${2:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
AGENT_CONFIG=${3:-${SCRIPT_DIR}/agent_config_localdocker.yaml}

export CUDA_VISIBLE_DEVICES=6,7

# ── Pre-flight checks ──
for var_name in DATA_PATH MODEL_PATH AGENT_CONFIG; do
    path="${!var_name}"
    if [ ! -e "$path" ]; then
        echo "ERROR: ${var_name} not found: ${path}" >&2
        exit 1
    fi
done

python ${PROJECT_ROOT}/examples/agent_interaction/parallel_infer.py \
    --data-path $DATA_PATH \
    --model-path $MODEL_PATH \
    --agent-config-path $AGENT_CONFIG \
    --num-workers 1 \
    --max-turns 100 \
    --tensor-parallel-size 1 \
    --n-gpus-per-node 2 \
    --prompt-length 4096 \
    --response-length 65536 \
    --max-samples 1 \
    --n 1
