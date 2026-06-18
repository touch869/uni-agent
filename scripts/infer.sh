#!/bin/bash
# Single-concurrency inference — minimal smoke test on 2 GPUs (1 sample).
# Can be executed from anywhere.
#
# For full multi-concurrency runs (8-GPU data-parallel, full dataset), use
# scripts/infer_multi.sh instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

MODEL_PATH=${1:-/path/to/Qwen3-4B}
DATA_PATH=${2:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
AGENT_CONFIG=${3:-${SCRIPT_DIR}/agent_config_localdocker.yaml}

# Optional: enable KVCAware llm-router (plugin_extension). Unset = built-in router.
# e.g. ROUTER_CONFIG=pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml
ROUTER_CONFIG=${ROUTER_CONFIG:-}

export CUDA_VISIBLE_DEVICES=6,7

# ── Pre-flight checks ──
for var_name in DATA_PATH MODEL_PATH AGENT_CONFIG; do
    path="${!var_name}"
    if [ ! -e "$path" ]; then
        echo "ERROR: ${var_name} not found: ${path}" >&2
        exit 1
    fi
done

ROUTER_ARGS=()
if [ -n "$ROUTER_CONFIG" ]; then
    ROUTER_ARGS+=(--router-config-path "$ROUTER_CONFIG")
fi

python ${PROJECT_ROOT}/examples/agent_interaction/parallel_infer.py \
    --data-path $DATA_PATH \
    --model-path $MODEL_PATH \
    --agent-config-path $AGENT_CONFIG \
    --num-workers 1 \
    --max-turns 100 \
    --tensor-parallel-size 1 \
    --n-gpus-per-node 2 \
    --prompt-length 4096 \
    --response-length 8192 \
    --max-samples 1 \
    --n 1 \
    "${ROUTER_ARGS[@]}"
