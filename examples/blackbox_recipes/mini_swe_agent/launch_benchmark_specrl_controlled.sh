#!/usr/bin/env bash

set -uo pipefail

# ==============================================================================
# User configuration
# Edit this section, then run:
#   bash examples/blackbox_recipes/mini_swe_agent/launch_benchmark_specrl_controlled.sh
# ==============================================================================

# GPU selection. Leave GPU_IDS empty to automatically select NUM_GPUS cards.
# To select physical cards manually, for example: GPU_IDS=(0 2 5 7)
NUM_GPUS=4
GPU_IDS=()
MAX_USED_MIB=1000
QUEUE_INTERVAL=10
STABLE_INTERVAL=3
STABLE_CHECKS=3

# Four-card separate-async layout: 2 trainer + 2 rollout.
TRAIN_GPUS=2
ROLLOUT_GPUS=2
TRAIN_TP=2
GEN_TP=2

# Model and data.
MODEL_PATH=/data1/zpy/workspace/models/Qwen3-1.7B
TRAIN_DATA=/data1/zpy/workspace/data/swe_agent/swe_rebench_filtered_openyuanrong.parquet
VAL_DATA=/data1/zpy/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet

# Benchmark size.
REPEATS=5
TOTAL_TRAINING_STEPS=30
TOTAL_EPOCHS=30
METRIC_WARMUP_STEPS=5
TRAIN_MAX_SAMPLES=1
BENCHMARK_SEED=1234

# Rollout and vLLM.
PROMPT_LENGTH=16384
RESPONSE_LENGTH=2048
MAX_NUM_BATCHED_TOKENS=4096
N=4
ROLLOUT_GPU_MEM_UTIL=0.45
MAX_NUM_SEQS=32
UPDATE_WEIGHTS_BUCKET_MB=128
ACTOR_PARAM_OFFLOAD=true
ASYNC_SCHEDULING=false

# Fixed-token performance mode. This is reproducible but normally produces
# zero reward because the agent gets only one action and cannot stop early.
AGENT_MAX_TURNS=1
IGNORE_EOS=true
FULL_DETERMINISM=true

# For a reward/correctness-oriented run, use these values instead:
# AGENT_MAX_TURNS=20
# IGNORE_EOS=false
# FULL_DETERMINISM=false

# Output. Leave BENCHMARK_TAG empty to use the benchmark's timestamped default.
BENCHMARK_TAG=
COMMAND_LOG=
DRY_RUN="${DRY_RUN:-false}"

# Additional NAME=VALUE entries passed to the benchmark.
EXTRA_ENV=(
  TRIAL_CLEANUP_SETTLE_SECONDS=30
)

# ==============================================================================
# Launcher implementation. No routine edits should be needed below this line.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
QUEUE_SCRIPT="${REPO_ROOT}/scripts/queue.sh"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_specrl_controlled.sh"

die() {
  printf '%s: error: %s\n' "${0##*/}" "$*" >&2
  exit 2
}

positive_int() {
  [[ $1 =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

nonnegative_int() {
  [[ $1 =~ ^[0-9]+$ ]]
}

boolean() {
  [[ $1 == true || $1 == false ]]
}

main() {
  local value gpu item
  local -a queue_args=() run_env=()

  (( $# == 0 )) || die "this launcher is configured by editing the file; command-line options are not accepted"
  [[ -x $QUEUE_SCRIPT ]] || die "queue script is not executable: $QUEUE_SCRIPT"
  [[ -f $BENCHMARK_SCRIPT ]] || die "benchmark not found: $BENCHMARK_SCRIPT"

  for value in "$NUM_GPUS" "$STABLE_CHECKS" "$TRAIN_GPUS" "$ROLLOUT_GPUS" \
    "$TRAIN_TP" "$GEN_TP" "$REPEATS" "$TOTAL_TRAINING_STEPS" "$TOTAL_EPOCHS" \
    "$PROMPT_LENGTH" "$RESPONSE_LENGTH" "$MAX_NUM_BATCHED_TOKENS" "$N" \
    "$AGENT_MAX_TURNS" "$TRAIN_MAX_SAMPLES" "$MAX_NUM_SEQS" \
    "$UPDATE_WEIGHTS_BUCKET_MB"; do
    positive_int "$value" || die "expected a positive integer, got: $value"
  done
  nonnegative_int "$MAX_USED_MIB" || die "MAX_USED_MIB must be a non-negative integer"
  nonnegative_int "$METRIC_WARMUP_STEPS" || die "METRIC_WARMUP_STEPS must be non-negative"
  nonnegative_int "$BENCHMARK_SEED" || die "BENCHMARK_SEED must be non-negative"
  boolean "$ACTOR_PARAM_OFFLOAD" || die "ACTOR_PARAM_OFFLOAD must be true or false"
  boolean "$IGNORE_EOS" || die "IGNORE_EOS must be true or false"
  boolean "$FULL_DETERMINISM" || die "FULL_DETERMINISM must be true or false"
  boolean "$ASYNC_SCHEDULING" || die "ASYNC_SCHEDULING must be true or false"
  boolean "$DRY_RUN" || die "DRY_RUN must be true or false"
  (( TOTAL_TRAINING_STEPS > METRIC_WARMUP_STEPS )) || \
    die "TOTAL_TRAINING_STEPS must exceed METRIC_WARMUP_STEPS"
  (( TRAIN_GPUS + ROLLOUT_GPUS == NUM_GPUS )) || \
    die "TRAIN_GPUS + ROLLOUT_GPUS must equal NUM_GPUS"
  (( TRAIN_GPUS % TRAIN_TP == 0 )) || die "TRAIN_GPUS must be divisible by TRAIN_TP"
  (( ROLLOUT_GPUS % GEN_TP == 0 )) || die "ROLLOUT_GPUS must be divisible by GEN_TP"
  [[ -f $MODEL_PATH/config.json ]] || die "invalid model path: $MODEL_PATH"
  [[ -f $TRAIN_DATA ]] || die "training data not found: $TRAIN_DATA"
  [[ -f $VAL_DATA ]] || die "validation data not found: $VAL_DATA"

  if (( ${#GPU_IDS[@]} > 0 )); then
    (( ${#GPU_IDS[@]} == NUM_GPUS )) || \
      die "GPU_IDS contains ${#GPU_IDS[@]} cards, expected NUM_GPUS=$NUM_GPUS"
    for gpu in "${GPU_IDS[@]}"; do
      nonnegative_int "$gpu" || die "invalid GPU ID: $gpu"
      queue_args+=(--gpu "$gpu")
    done
  else
    queue_args+=(--num-gpus "$NUM_GPUS")
  fi
  queue_args+=(
    --max-used-mib "$MAX_USED_MIB"
    --interval "$QUEUE_INTERVAL"
    --stable-interval "$STABLE_INTERVAL"
    --stable-checks "$STABLE_CHECKS"
    --workdir "$REPO_ROOT"
  )
  [[ -z $COMMAND_LOG ]] || queue_args+=(--log "$COMMAND_LOG")
  $DRY_RUN && queue_args+=(--dry-run)

  run_env=(
    "MODEL_PATH=$MODEL_PATH"
    "TRAIN_DATA=$TRAIN_DATA"
    "VAL_DATA=$VAL_DATA"
    "REPEATS=$REPEATS"
    "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS"
    "TOTAL_EPOCHS=$TOTAL_EPOCHS"
    "METRIC_WARMUP_STEPS=$METRIC_WARMUP_STEPS"
    "PROMPT_LENGTH=$PROMPT_LENGTH"
    "RESPONSE_LENGTH=$RESPONSE_LENGTH"
    "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    "N=$N"
    "AGENT_MAX_TURNS=$AGENT_MAX_TURNS"
    "TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES"
    "TRAIN_NGPUS_PER_NODE=$TRAIN_GPUS"
    "ROLLOUT_NGPUS_PER_NODE=$ROLLOUT_GPUS"
    "TRAIN_TP=$TRAIN_TP"
    "GEN_TP=$GEN_TP"
    "ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL"
    "MAX_NUM_SEQS=$MAX_NUM_SEQS"
    "UPDATE_WEIGHTS_BUCKET_MB=$UPDATE_WEIGHTS_BUCKET_MB"
    "ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD"
    "IGNORE_EOS=$IGNORE_EOS"
    "FULL_DETERMINISM=$FULL_DETERMINISM"
    "ASYNC_SCHEDULING=$ASYNC_SCHEDULING"
    "BENCHMARK_SEED=$BENCHMARK_SEED"
  )
  [[ -z $BENCHMARK_TAG ]] || run_env+=("BENCHMARK_TAG=$BENCHMARK_TAG")
  for item in "${EXTRA_ENV[@]}"; do
    [[ $item == *=* && $item != =* ]] || die "invalid EXTRA_ENV entry: $item"
    run_env+=("$item")
  done

  exec "$QUEUE_SCRIPT" "${queue_args[@]}" -- \
    env "${run_env[@]}" bash "$BENCHMARK_SCRIPT"
}

main "$@"
