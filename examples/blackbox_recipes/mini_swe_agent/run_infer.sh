#!/usr/bin/env bash
# Standalone inference for the blackbox mini-swe-agent recipe.
# Runs rollout + reward only (no Megatron trainer) and reports resolve rate.
#
# Usage:
#   bash examples/blackbox_recipes/mini_swe_agent/run_infer.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ─────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH="${HOME}/models/Qwen3.5-9B"
if [[ ! -f "${DEFAULT_MODEL_PATH}/config.json" && -f "${WORKSPACE_ROOT}/models/SWE-Lego-Qwen3-8B/config.json" ]]; then
    DEFAULT_MODEL_PATH="${WORKSPACE_ROOT}/models/SWE-Lego-Qwen3-8B"
fi
DEFAULT_DATA_PATH="${HOME}/data/swe_agent/swe_bench_verified.parquet"
if [[ ! -f "${DEFAULT_DATA_PATH}" && -f "${WORKSPACE_ROOT}/data/swe_agent/swe_bench_verified_openyuanrong.parquet" ]]; then
    DEFAULT_DATA_PATH="${WORKSPACE_ROOT}/data/swe_agent/swe_bench_verified_openyuanrong.parquet"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: invalid MODEL_PATH (missing config.json): ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH does not exist: ${DATA_PATH}" >&2
    exit 2
fi

# ── Inference parameters ─────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
# Keep standalone inference within the KV-cache capacity of the documented
# 2x24GB setup, including long 100-turn tool trajectories. Individual model
# calls are capped separately by the gateway.
RESPONSE_LENGTH="${RESPONSE_LENGTH:-28672}"
MODEL_CONTEXT_HEADROOM="${MODEL_CONTEXT_HEADROOM:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
N="${N:-1}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-2}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-8}"

# GPU 3 is excluded because it has previously entered a persistent CUDA
# launch-failure state. Use a known healthy pair by default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
IFS=, read -ra VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
for GPU_ID in "${VISIBLE_GPU_IDS[@]}"; do
    if [[ "${GPU_ID//[[:space:]]/}" == "3" ]]; then
        echo "ERROR: GPU 3 is disabled for Uni-Agent inference because it is unstable." >&2
        echo "Choose another pair, for example CUDA_VISIBLE_DEVICES=4,5." >&2
        exit 2
    fi
done

# ── Agent parameters ─────────────────────────────────────────────────────
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}"
SWE_AGENT_TOOL_IMAGE="${SWE_AGENT_TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest}"
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"

# ── AKernel (remote sandbox) ─────────────────────────────────────────────
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-${OPENYUANRONG_SERVER_ADDRESS:-}}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-${OPENYUANRONG_TOKEN:-}}"
export AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"
if [[ -z "${AKERNEL_SERVER_ADDRESS}" || -z "${AKERNEL_TOKEN}" ]]; then
    echo "ERROR: AKERNEL_SERVER_ADDRESS/OPENYUANRONG_SERVER_ADDRESS and AKERNEL_TOKEN/OPENYUANRONG_TOKEN must be set for the OpenYuanRong remote sandbox." >&2
    exit 2
fi

# ── Logging & env ────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
export AGENT_MAX_TURNS
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

CUDA_DEVICE_COUNT="$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || true)"
if [[ ! "${CUDA_DEVICE_COUNT}" =~ ^[0-9]+$ ]] || (( CUDA_DEVICE_COUNT < TP )); then
    echo "ERROR: CUDA preflight found ${CUDA_DEVICE_COUNT:-0} usable device(s), but TP=${TP}." >&2
    echo "Check nvidia-smi and CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; a GPU in Unknown Error state requires a host GPU reset or reboot." >&2
    exit 2
fi

echo "=== Mini-SWE-Agent Blackbox Inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Engine:      ${ENGINE} (TP=${TP}, GPUs/node=${N_GPUS_PER_NODE})"
echo "Visible GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "Context:     headroom=${MODEL_CONTEXT_HEADROOM}"
echo "Turns:       ${AGENT_MAX_TURNS}"
echo "Tool image:  ${SWE_AGENT_TOOL_IMAGE}"
echo "Batch:       n=${N}, gateway=${GATEWAY_COUNT}, max_sessions=${MAX_CONCURRENT_SESSIONS}"
if [[ -n "${GATEWAY_MESSAGE_JSONL_PATH:-}" ]]; then
    echo "Messages:    ${GATEWAY_MESSAGE_JSONL_PATH}"
fi
echo "========================================="

python examples/blackbox_recipes/mini_swe_agent/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --model-context-headroom "${MODEL_CONTEXT_HEADROOM}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --max-concurrent-sessions "${MAX_CONCURRENT_SESSIONS}" \
    --tool-image "${SWE_AGENT_TOOL_IMAGE}" \
    --run-timeout "${SWE_AGENT_RUN_TIMEOUT}" \
    --max-turns "${AGENT_MAX_TURNS}"
