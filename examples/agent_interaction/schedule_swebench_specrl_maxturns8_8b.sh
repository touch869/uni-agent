#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke)
    MAX_SAMPLES=2
    RUN_TAG="qwen3_8b_smoke_maxturns8_2_n2_tp2_gpu085"
    ;;
  formal)
    MAX_SAMPLES=64
    RUN_TAG="qwen3_8b_formal_maxturns8_64_n2_tp2_gpu085"
    ;;
  *)
    echo "usage: $0 {smoke|formal}" >&2
    exit 2
    ;;
esac

OUT_DIR="/data1/zpy/workspace/output/swebench_eval"
CONTAINER_OUT_DIR="/workspace/output/swebench_eval"
LOG_PATH="${OUT_DIR}/${RUN_TAG}.log"
mkdir -p "${OUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG_PATH}"
}

run_in_container() {
  local container="$1"
  shift
  log "start container=${container} phase=$1"
  local phase="$1"
  shift
  docker exec "${container}" bash -lc "$*" 2>&1 | tee -a "${LOG_PATH}"
  log "finish container=${container} phase=${phase}"
}

# Keep cache-server process discovery and termination separate from startup. In
# particular, never put the server script name and pkill -f in one bash -lc.
stop_cache_server() {
  local pids
  pids="$(docker exec uniagent-specrl pgrep -f '[s]tart_specrl_cache_server.py' || true)"
  if [[ -n "${pids}" ]]; then
    log "stopping old cache server pid(s): ${pids//$'\n'/,}"
    while read -r pid; do
      [[ -n "${pid}" ]] && docker exec uniagent-specrl kill "${pid}"
    done <<< "${pids}"
  fi
}

COMMON_ENV="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1; cd /workspace/uni-agent"
SPECRL_ENV="${COMMON_ENV}; export VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0"
BASE_ARGS="--data-path /workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet --model-path /workspace/models/Qwen3-8B-base --agent-config-path /workspace/uni-agent/examples/agent_interaction/agent_config_local.yaml --engine vllm --max-samples ${MAX_SAMPLES} --max-turns 8 --prompt-length 4096 --response-length 2048 --temperature 0.8 --top-p 0.9 --n 2 --num-workers 1 --n-gpus-per-node 2 --tensor-parallel-size 2 --gpu-memory-utilization 0.85"

log "experiment_begin mode=${MODE} tag=${RUN_TAG}"
run_in_container uniagent-eval vanilla "${COMMON_ENV}; ray stop --force || true; python3 examples/agent_interaction/parallel_infer.py ${BASE_ARGS} --result-path ${CONTAINER_OUT_DIR}/vanilla_${RUN_TAG}.json; ray stop --force || true"

stop_cache_server
docker exec uniagent-specrl rm -rf /dev/shm/SUFFIX_CACHE
run_in_container uniagent-specrl cache_start "${SPECRL_ENV}; nohup python3 -u examples/agent_interaction/start_specrl_cache_server.py > ${CONTAINER_OUT_DIR}/specrl_cache_${RUN_TAG}.log 2>&1 & sleep 5; python3 examples/agent_interaction/check_specrl_ready.py; python3 -c 'import socket; s=socket.create_connection((\"127.0.0.1\",6378),timeout=5); s.close(); print(\"cache_server_tcp_ready=127.0.0.1:6378\")'"
run_in_container uniagent-specrl specrl_warm "${SPECRL_ENV}; ray stop --force || true; python3 examples/agent_interaction/parallel_infer.py ${BASE_ARGS} --enable-specrl --specrl-update-cache --result-path ${CONTAINER_OUT_DIR}/specrl_warm_${RUN_TAG}.json; ray stop --force || true"

ROUNDS=3
[[ "${MODE}" == "smoke" ]] && ROUNDS=1
for round in $(seq 1 "${ROUNDS}"); do
  run_in_container uniagent-specrl "specrl_cached_round${round}" "${SPECRL_ENV}; ray stop --force || true; python3 examples/agent_interaction/parallel_infer.py ${BASE_ARGS} --enable-specrl --result-path ${CONTAINER_OUT_DIR}/specrl_cached_${RUN_TAG}_round${round}.json; ray stop --force || true"
done

run_in_container uniagent-specrl summary "cd /workspace/uni-agent; python3 examples/agent_interaction/summarize_swebench_8b.py --tag ${RUN_TAG}"
log "experiment_done mode=${MODE} tag=${RUN_TAG}"
