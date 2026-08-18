#!/usr/bin/env bash
# Controlled paired baseline/SPEC-RL benchmark with fixed-length responses.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_TRAIN="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/run_train.sh"
ANALYZER="${REPO_ROOT}/examples/blackbox_recipes/mini_swe_agent/analyze_specrl_benchmark.py"

MODEL_PATH="${MODEL_PATH:-/data1/zpy/workspace/models/Qwen3-1.7B}"
TRAIN_DATA="${TRAIN_DATA:-/data1/zpy/workspace/data/swe_agent/swe_rebench_filtered_openyuanrong.parquet}"
VAL_DATA="${VAL_DATA:-/data1/zpy/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet}"

REPEATS="${REPEATS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${TOTAL_TRAINING_STEPS}}"
METRIC_WARMUP_STEPS="${METRIC_WARMUP_STEPS:-5}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
N="${N:-4}"
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1}"
BENCHMARK_SEED="${BENCHMARK_SEED:-1234}"
TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-4}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-4}"
TRAIN_TP="${TRAIN_TP:-2}"
GEN_TP="${GEN_TP:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-128}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
IGNORE_EOS="${IGNORE_EOS:-true}"
FULL_DETERMINISM="${FULL_DETERMINISM:-true}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"
TRIAL_CLEANUP_SETTLE_SECONDS="${TRIAL_CLEANUP_SETTLE_SECONDS:-30}"
BENCHMARK_TAG="${BENCHMARK_TAG:-specrl_controlled_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${REPO_ROOT}/outputs/specrl_benchmark/${BENCHMARK_TAG}}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    pkill -TERM -f 'VLLM::EngineCore' >/dev/null 2>&1 || true
    local attempt
    for ((attempt = 0; attempt < 60; attempt++)); do
        if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
            break
        fi
        sleep 1
    done
    sleep "${TRIAL_CLEANUP_SETTLE_SECONDS}"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

validate_inputs() {
    local visible_gpu_count total_required_gpus

    command -v ray >/dev/null 2>&1 || fail "ray is not available"
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available"
    [[ -f "${MODEL_PATH}/config.json" ]] || fail "invalid model directory: ${MODEL_PATH}"
    [[ -f "${TRAIN_DATA}" ]] || fail "missing train data: ${TRAIN_DATA}"
    [[ -f "${VAL_DATA}" ]] || fail "missing validation data: ${VAL_DATA}"
    [[ -x "${RUN_TRAIN}" || -f "${RUN_TRAIN}" ]] || fail "missing run_train.sh: ${RUN_TRAIN}"
    [[ -f "${ANALYZER}" ]] || fail "missing analyzer: ${ANALYZER}"
    [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS must be a positive integer"
    [[ "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "TOTAL_TRAINING_STEPS must be a positive integer"
    [[ "${TOTAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || fail "TOTAL_EPOCHS must be a positive integer"
    [[ "${METRIC_WARMUP_STEPS}" =~ ^[0-9]+$ ]] || fail "METRIC_WARMUP_STEPS must be a non-negative integer"
    [[ "${PROMPT_LENGTH}" =~ ^[1-9][0-9]*$ ]] || fail "PROMPT_LENGTH must be a positive integer"
    [[ "${RESPONSE_LENGTH}" =~ ^[1-9][0-9]*$ ]] || fail "RESPONSE_LENGTH must be a positive integer"
    [[ "${MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_NUM_BATCHED_TOKENS must be a positive integer"
    [[ "${N}" =~ ^[1-9][0-9]*$ ]] || fail "N must be a positive integer"
    [[ "${AGENT_MAX_TURNS}" =~ ^[1-9][0-9]*$ ]] || fail "AGENT_MAX_TURNS must be a positive integer"
    [[ "${TRAIN_MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_MAX_SAMPLES must be a positive integer"
    (( REPEATS >= 1 )) || fail "REPEATS must be >= 1"
    (( TOTAL_TRAINING_STEPS > METRIC_WARMUP_STEPS )) || \
        fail "TOTAL_TRAINING_STEPS must be greater than METRIC_WARMUP_STEPS"
    [[ "${TRAIN_NGPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_NGPUS_PER_NODE must be a positive integer"
    [[ "${ROLLOUT_NGPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || fail "ROLLOUT_NGPUS_PER_NODE must be a positive integer"
    [[ "${TRAIN_TP}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_TP must be a positive integer"
    [[ "${GEN_TP}" =~ ^[1-9][0-9]*$ ]] || fail "GEN_TP must be a positive integer"
    (( TRAIN_NGPUS_PER_NODE % TRAIN_TP == 0 )) || \
        fail "TRAIN_NGPUS_PER_NODE must be divisible by TRAIN_TP"
    (( ROLLOUT_NGPUS_PER_NODE % GEN_TP == 0 )) || \
        fail "ROLLOUT_NGPUS_PER_NODE must be divisible by GEN_TP"
    visible_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
    total_required_gpus=$((TRAIN_NGPUS_PER_NODE + ROLLOUT_NGPUS_PER_NODE))
    (( visible_gpu_count >= total_required_gpus )) || \
        fail "benchmark requires ${total_required_gpus} visible GPUs (${TRAIN_NGPUS_PER_NODE} trainer + ${ROLLOUT_NGPUS_PER_NODE} rollout), but only ${visible_gpu_count} are visible"
    awk -v value="${ROLLOUT_GPU_MEM_UTIL}" \
        'BEGIN {exit !(value > 0 && value <= 1)}' || \
        fail "ROLLOUT_GPU_MEM_UTIL must be greater than 0 and at most 1"
    [[ "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_NUM_SEQS must be a positive integer"
    [[ "${UPDATE_WEIGHTS_BUCKET_MB}" =~ ^[1-9][0-9]*$ ]] || fail "UPDATE_WEIGHTS_BUCKET_MB must be a positive integer"
    [[ "${ACTOR_PARAM_OFFLOAD}" == "true" || "${ACTOR_PARAM_OFFLOAD}" == "false" ]] || fail "ACTOR_PARAM_OFFLOAD must be true or false"
    [[ "${IGNORE_EOS}" == "true" || "${IGNORE_EOS}" == "false" ]] || fail "IGNORE_EOS must be true or false"
    [[ "${FULL_DETERMINISM}" == "true" || "${FULL_DETERMINISM}" == "false" ]] || fail "FULL_DETERMINISM must be true or false"
    [[ "${ASYNC_SCHEDULING}" == "true" || "${ASYNC_SCHEDULING}" == "false" ]] || fail "ASYNC_SCHEDULING must be true or false"
    [[ "${TRIAL_CLEANUP_SETTLE_SECONDS}" =~ ^[0-9]+$ ]] || fail "TRIAL_CLEANUP_SETTLE_SECONDS must be a non-negative integer"
}

run_trial() {
    local mode="$1"
    local repeat="$2"
    local enabled=false
    local trial_name trial_dir start_ns end_ns exit_code wall_seconds

    [[ "${mode}" == "specrl" ]] && enabled=true
    trial_name="$(printf 'repeat_%02d_%s' "${repeat}" "${mode}")"
    trial_dir="${BENCHMARK_DIR}/${trial_name}"
    mkdir -p "${trial_dir}"

    cleanup
    start_ns="$(date +%s%N)"
    echo "START ${trial_name} $(date --iso-8601=seconds)"

    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_DATA="${TRAIN_DATA}" \
    VAL_DATA="${VAL_DATA}" \
    NNODES=1 \
    N_GPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE}" \
    ROLLOUT_NNODES=1 \
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE}" \
    TRAIN_TP="${TRAIN_TP}" \
    GEN_TP="${GEN_TP}" \
    SPECRL_ENABLED="${enabled}" \
    SPECRL_SEED="${BENCHMARK_SEED}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
    PROMPT_LENGTH="${PROMPT_LENGTH}" \
    RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES}" \
    VAL_MAX_SAMPLES=1 \
    TRAIN_BATCH_SIZE=1 \
    VAL_BATCH_SIZE=1 \
    PPO_MINI_BATCH_SIZE=1 \
    PARAMETER_SYNC_STEP=1 \
    NUM_WARMUP_BATCHES=0 \
    SEPARATE_NUM_WARMUP_BATCHES=0 \
    N="${N}" \
    AGENT_MAX_TURNS="${AGENT_MAX_TURNS}" \
    UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB}" \
    TORCH_CUDA_ARCH_LIST=8.6 \
    VANILLA_MBRIDGE=True \
    RAY_SUBMIT_MODE=local \
    VAL_BEFORE_TRAIN=false \
    TEST_FREQ=-1 \
    SAVE_FREQ=-1 \
    PROJECT_NAME=specrl_controlled \
    EXPERIMENT_NAME="${trial_name}" \
    CKPTS_DIR="${trial_dir}/checkpoints" \
    VERL_FILE_LOGGER_PATH="${trial_dir}/metrics.jsonl" \
    bash "${RUN_TRAIN}" \
        "trainer.logger=[console,file]" \
        "data.seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.rollout.seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.actor.data_loader_seed=${BENCHMARK_SEED}" \
        "actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD}" \
        "actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_PARAM_OFFLOAD}" \
        "actor_rollout_ref.rollout.calculate_log_probs=true" \
        "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}" \
        "actor_rollout_ref.rollout.ignore_eos=${IGNORE_EOS}" \
        "actor_rollout_ref.rollout.full_determinism=${FULL_DETERMINISM}" \
        "actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=${ASYNC_SCHEDULING}" \
        "~actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn" \
        "~actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm" \
        2>&1 | tee "${trial_dir}/console.log"
    exit_code=${PIPESTATUS[0]}

    end_ns="$(date +%s%N)"
    wall_seconds="$(
        awk -v start="${start_ns}" -v end="${end_ns}" \
            'BEGIN {printf "%.9f", (end-start)/1000000000}'
    )"
    printf \
        '{"mode":"%s","repeat":%d,"start_ns":%s,"end_ns":%s,"wall_seconds":%s,"exit_code":%d}\n' \
        "${mode}" "${repeat}" "${start_ns}" "${end_ns}" "${wall_seconds}" "${exit_code}" \
        >"${trial_dir}/wall.json"

    echo "END ${trial_name} exit=${exit_code} wall=${wall_seconds}s"
    if (( exit_code != 0 )); then
        tail -120 "${trial_dir}/console.log"
        return "${exit_code}"
    fi
    if [[ ! -s "${trial_dir}/metrics.jsonl" ]]; then
        echo "Missing metrics: ${trial_dir}/metrics.jsonl" >&2
        return 1
    fi
}

main() {
    local repeat

    validate_inputs
    mkdir -p "${BENCHMARK_DIR}"
    trap cleanup EXIT INT TERM

    cleanup
    echo "Benchmark output: ${BENCHMARK_DIR}"
    echo "Run config: repeats=${REPEATS}, steps=${TOTAL_TRAINING_STEPS}, epochs=${TOTAL_EPOCHS}, metric_warmup=${METRIC_WARMUP_STEPS}"
    echo "Sampling: n=${N}, prompt_length=${PROMPT_LENGTH}, response_length=${RESPONSE_LENGTH}, max_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, train_max_samples=${TRAIN_MAX_SAMPLES}"
    echo "Agent: max_turns=${AGENT_MAX_TURNS}, ignore_eos=${IGNORE_EOS}, full_determinism=${FULL_DETERMINISM}, async_scheduling=${ASYNC_SCHEDULING}"
    echo "GPU layout: trainer=${TRAIN_NGPUS_PER_NODE} (TP=${TRAIN_TP}), rollout=${ROLLOUT_NGPUS_PER_NODE} (TP=${GEN_TP})"
    echo "Rollout GPU memory utilization: ${ROLLOUT_GPU_MEM_UTIL}"
    echo "Rollout max sequences: ${MAX_NUM_SEQS}"
    echo "Weight sync bucket: ${UPDATE_WEIGHTS_BUCKET_MB} MiB"
    echo "Actor parameter offload: ${ACTOR_PARAM_OFFLOAD}"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader

    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        if (( repeat % 2 == 1 )); then
            run_trial baseline "${repeat}" || return $?
            run_trial specrl "${repeat}" || return $?
        else
            run_trial specrl "${repeat}" || return $?
            run_trial baseline "${repeat}" || return $?
        fi
    done

    cleanup
    python3 "${ANALYZER}" "${BENCHMARK_DIR}" --warmup-steps "${METRIC_WARMUP_STEPS}" \
        | tee "${BENCHMARK_DIR}/analyzer.log"
    local analyzer_exit=${PIPESTATUS[0]}
    echo "Benchmark output: ${BENCHMARK_DIR}"
    return "${analyzer_exit}"
}

main "$@"
