#!/usr/bin/env bash
set -euo pipefail

STAMP="${RUN_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
OUT_HOST="/data1/zpy/workspace/output/swebench_eval"
OUT="/workspace/output/swebench_eval"
REPO="/workspace/uni-agent"
PYTHON="/opt/uni-agent-venv/bin/python"
DATA="/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet"
MODEL="/workspace/models/Qwen3.5-9B"
CONFIG="${REPO}/examples/agent_interaction/agent_config_qwen35_evidence.yaml"
EVIDENCE_DIR="${OUT}/qwen35_9b_trajectories"
LOG="${OUT_HOST}/qwen35_9b_two_runs_${STAMP}.log"
CACHE_LOG="${OUT}/qwen35_9b_cache_${STAMP}.log"
COMMON_ENV="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 LOG_FLUSH_EACH_LINE=1 LD_LIBRARY_PATH=/opt/specrl-libs:\${LD_LIBRARY_PATH:-}; cd ${REPO}"
mkdir -p "${OUT_HOST}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG}"
}

stop_cache_server() {
  local pids
  pids="$(docker exec uniagent-eval pgrep -f '[s]tart_specrl_cache_server.py' || true)"
  if [[ -n "${pids}" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] && docker exec uniagent-eval kill "${pid}"
    done <<< "${pids}"
  fi
}

start_clean_cache_server() {
  stop_cache_server
  docker exec uniagent-eval rm -rf /dev/shm/SUFFIX_CACHE
  docker exec uniagent-eval bash -lc "${COMMON_ENV}; nohup ${PYTHON} -u examples/agent_interaction/start_specrl_cache_server.py > '${CACHE_LOG}' 2>&1 &"
  sleep 5
  docker exec uniagent-eval bash -lc "${COMMON_ENV}; ${PYTHON} -c 'import importlib.util,vllm; assert vllm.__version__.startswith(\"0.17.\"); assert importlib.util.find_spec(\"specrl.suffix_cache\"); assert importlib.util.find_spec(\"uni_agent.histospec_vllm017_plugin\"); print(\"histospec_runtime_ready\",vllm.__version__)'; ${PYTHON} -c 'import socket; s=socket.create_connection((\"127.0.0.1\",6378),timeout=5); s.close(); print(\"cache_server_ready\")'" | tee -a "${LOG}"
}

run_one() {
  local run_name="$1" extra_flags="$2"
  log "start run=${run_name}"
  docker exec uniagent-eval bash -lc "${COMMON_ENV}; ray stop --force || true; ${PYTHON} examples/agent_interaction/parallel_infer_swe_lego_evidence.py \
    --run-name '${run_name}' --data-path '${DATA}' --model-path '${MODEL}' \
    --agent-config-path '${CONFIG}' --result-path '${OUT}/${run_name}.json' \
    --evidence-dir '${EVIDENCE_DIR}' --evidence-jsonl '${OUT}/${run_name}_evidence.jsonl' \
    --max-samples 500 --n 1 --max-turns 100 --prompt-length 8192 --response-length 57344 \
    --temperature 1.0 --top-p 0.7 --top-k -1 --seed 31001 --enforce-eager \
    --num-workers 1 --n-gpus-per-node 2 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --specrl-num-speculative-tokens 5 ${extra_flags}; ray stop --force || true" 2>&1 | tee -a "${LOG}"
  log "finish run=${run_name}"
}

vanilla="qwen35_9b_vanilla_${STAMP}"
histospec="qwen35_9b_histospec_${STAMP}"
start_clean_cache_server
docker exec uniagent-eval bash -lc "cd ${REPO}; ${PYTHON} examples/agent_interaction/prepare_swe_lego_images.py --data-path '${DATA}' --max-samples 500 --manifest '${OUT}/qwen35_9b_image_manifest_${STAMP}.json'" 2>&1 | tee -a "${LOG}"
run_one "${vanilla}" "--specrl-update-cache"
run_one "${histospec}" "--enable-specrl"
log "complete vanilla=${vanilla} histospec=${histospec}"
