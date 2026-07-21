#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/openyuanrong.env}"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-1.7B}"
export DATA_PATH="${DATA_PATH:-/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet}"
export MAX_SAMPLES="${MAX_SAMPLES:-1}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
export TP="${TP:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-1}"
export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"

bash "${SCRIPT_DIR}/check_mini_swe_env.sh"
bash "${SCRIPT_DIR}/run_mini_swe.sh"
