#!/usr/bin/env bash
# Paired wall-clock and trainer-metric benchmark for baseline vs SPEC-RL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_TRAIN="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/run_train.sh"
ANALYZER="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/analyze_specrl_benchmark.py"

BENCHMARK_TAG="${BENCHMARK_TAG:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${REPO_ROOT}/outputs/specrl_benchmark/${BENCHMARK_TAG}}"
REPEATS="${REPEATS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-6}"
METRIC_WARMUP_STEPS="${METRIC_WARMUP_STEPS:-2}"
BENCHMARK_SEED="${BENCHMARK_SEED:-1234}"
RESET_RAY_BETWEEN_RUNS="${RESET_RAY_BETWEEN_RUNS:-true}"
BENCHMARK_VAL_BEFORE_TRAIN="${BENCHMARK_VAL_BEFORE_TRAIN:-false}"
BENCHMARK_TEST_FREQ="${BENCHMARK_TEST_FREQ:--1}"
BENCHMARK_SAVE_FREQ="${BENCHMARK_SAVE_FREQ:--1}"

# Small repeated dataset defaults are deliberate: SPEC-RL needs the same model
# context to reappear after a policy update before the cache can produce a hit.
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
SEPARATE_NUM_WARMUP_BATCHES="${SEPARATE_NUM_WARMUP_BATCHES:-1}"
N="${N:-2}"
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-8}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
SPECRL_BIAS="${SPECRL_BIAS:-0.5}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.35}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-512}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
VANILLA_MBRIDGE="${VANILLA_MBRIDGE:-True}"

MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"

if ! command -v ray >/dev/null 2>&1; then
    echo "ray is not available in this environment; run the benchmark in the verl training container" >&2
    exit 127
fi
if ! python3 -c 'import transfer_queue' >/dev/null 2>&1; then
    echo "Python package TransferQueue is required; install it with: python3 -m pip install TransferQueue==0.1.8" >&2
    exit 2
fi
if ! python3 -c 'from pathlib import Path; import akernel_sdk; assert "wss://" in Path(akernel_sdk.__file__).with_name("sandbox_api.py").read_text()' >/dev/null 2>&1; then
    echo "akernel-sdk must use wss:// tunnels for the configured HTTPS/443 gateway" >&2
    exit 2
fi
if ! python3 -c 'from mbridge import AutoBridge' >/dev/null 2>&1; then
    echo "Legacy mbridge is required for this benchmark environment" >&2
    exit 2
fi
if ! python3 -c 'from datasets.features.features import _FEATURE_TYPES; assert "List" in _FEATURE_TYPES' >/dev/null 2>&1; then
    echo "datasets with List feature support is required; install it with: python3 -m pip install datasets==5.0.0" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "MODEL_PATH is not a model directory (missing config.json): ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2
    exit 2
fi
if [[ ! -f "${VAL_DATA}" ]]; then
    echo "VAL_DATA does not exist: ${VAL_DATA}" >&2
    exit 2
fi

if (( REPEATS < 1 )); then
    echo "REPEATS must be >= 1" >&2
    exit 2
fi
if (( TOTAL_TRAINING_STEPS <= METRIC_WARMUP_STEPS )); then
    echo "TOTAL_TRAINING_STEPS must be greater than METRIC_WARMUP_STEPS" >&2
    exit 2
fi
if [[ "${TRAIN_BATCH_SIZE}" -ne $((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE)) ]]; then
    echo "TRAIN_BATCH_SIZE must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE for separate_async" >&2
    exit 2
fi

mkdir -p "${BENCHMARK_DIR}"

reset_ray() {
    if [[ "${RESET_RAY_BETWEEN_RUNS}" == "true" ]] && command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}

run_trial() {
    local mode="$1"
    local repeat="$2"
    local enabled="false"
    [[ "${mode}" == "specrl" ]] && enabled="true"

    local trial_name
    trial_name="$(printf 'repeat_%02d_%s' "${repeat}" "${mode}")"
    local trial_dir="${BENCHMARK_DIR}/${trial_name}"
    local metrics_path="${trial_dir}/metrics.jsonl"
    local console_path="${trial_dir}/console.log"
    local checkpoint_dir="${trial_dir}/checkpoints"
    mkdir -p "${trial_dir}"

    reset_ray
    local start_ns end_ns exit_code
    start_ns="$(date +%s%N)"
    echo "Starting ${trial_name} at $(date --iso-8601=seconds)"

    set +e
    VERL_FILE_LOGGER_PATH="${metrics_path}" \
    RAY_SUBMIT_MODE=local \
    SPECRL_ENABLED="${enabled}" \
    SPECRL_BIAS="${SPECRL_BIAS}" \
    ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
    UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    VANILLA_MBRIDGE="${VANILLA_MBRIDGE}" \
    TEMPERATURE=1.0 \
    TOP_P=1.0 \
    TOP_K=-1 \
    N="${N}" \
    AGENT_MAX_TURNS="${AGENT_MAX_TURNS}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    PROMPT_LENGTH="${PROMPT_LENGTH}" \
    RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES}" \
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
    PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP}" \
    NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES}" \
    SEPARATE_NUM_WARMUP_BATCHES="${SEPARATE_NUM_WARMUP_BATCHES}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    TOTAL_EPOCHS=100 \
    VAL_BEFORE_TRAIN="${BENCHMARK_VAL_BEFORE_TRAIN}" \
    TEST_FREQ="${BENCHMARK_TEST_FREQ}" \
    SAVE_FREQ="${BENCHMARK_SAVE_FREQ}" \
    PROJECT_NAME=specrl_benchmark \
    EXPERIMENT_NAME="${trial_name}" \
    CKPTS_DIR="${checkpoint_dir}" \
    bash "${RUN_TRAIN}" \
        "trainer.logger=['console','file']" \
        "data.seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.rollout.seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.actor.data_loader_seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.rollout.calculate_log_probs=true" \
        "~actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn" \
        "~actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm" \
        2>&1 | tee "${console_path}"
    exit_code="${PIPESTATUS[0]}"
    set -e

    end_ns="$(date +%s%N)"
    local wall_seconds
    wall_seconds="$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN { printf "%.9f", (end-start)/1000000000 }')"
    printf '{"mode":"%s","repeat":%d,"start_ns":%s,"end_ns":%s,"wall_seconds":%s,"exit_code":%d}\n' \
        "${mode}" "${repeat}" "${start_ns}" "${end_ns}" "${wall_seconds}" "${exit_code}" \
        > "${trial_dir}/wall.json"

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "${trial_name} failed with exit code ${exit_code}; see ${console_path}" >&2
        exit "${exit_code}"
    fi
    if [[ ! -s "${metrics_path}" ]]; then
        echo "${trial_name} produced no file-logger metrics at ${metrics_path}" >&2
        exit 1
    fi
    echo "Finished ${trial_name}: ${wall_seconds}s"
}

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    # Alternate order across repeats to reduce first-run/cache/thermal bias.
    if (( repeat % 2 == 1 )); then
        run_trial baseline "${repeat}"
        run_trial specrl "${repeat}"
    else
        run_trial specrl "${repeat}"
        run_trial baseline "${repeat}"
    fi
done

reset_ray
python3 "${ANALYZER}" "${BENCHMARK_DIR}" --warmup-steps "${METRIC_WARMUP_STEPS}"
