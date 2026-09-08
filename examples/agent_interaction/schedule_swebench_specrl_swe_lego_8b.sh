#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke_vanilla}"
STAMP="${RUN_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
OUT_HOST="/data1/zpy/workspace/output/swebench_eval"
OUT="/workspace/output/swebench_eval"
REPO="/workspace/uni-agent"
DATA="/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet"
MODEL="/workspace/models/SWE-Lego-Qwen3-8B"
CONFIG="/workspace/uni-agent/examples/agent_interaction/agent_config_swe_lego_evidence.yaml"
EVIDENCE_DIR="${OUT}/swe_lego_qwen3_8b_trajectories"
LOG="${OUT_HOST}/swe_lego_qwen3_8b_${MODE}_${STAMP}.log"
mkdir -p "${OUT_HOST}"

COMMON_ENV="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 LOG_FLUSH_EACH_LINE=1; cd ${REPO}"
SPECRL_ENV="${COMMON_ENV}; export VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG}"
}

run_one() {
  local container="$1" run_name="$2" max_samples="$3" n="$4" max_turns="$5"
  local prompt_length="$6" response_length="$7" temperature="$8" top_p="$9" top_k="${10}"
  local specrl_flags="${11:-}" env_cmd
  [[ "${container}" == "uniagent-specrl" ]] && env_cmd="${SPECRL_ENV}" || env_cmd="${COMMON_ENV}"
  local result="${OUT}/${run_name}.json"
  local evidence_jsonl="${OUT}/${run_name}_evidence.jsonl"
  docker exec "${container}" bash -lc "cd ${REPO}; python3 examples/agent_interaction/prepare_swe_lego_images.py \
    --data-path '${DATA}' --max-samples '${max_samples}' \
    --manifest '${OUT}/${run_name}_image_manifest.json'" 2>&1 | tee -a "${LOG}"
  log "start run=${run_name} container=${container}"
  docker exec "${container}" bash -lc "${env_cmd}; ray stop --force || true; python3 examples/agent_interaction/parallel_infer_swe_lego_evidence.py \
    --run-name '${run_name}' --data-path '${DATA}' --model-path '${MODEL}' \
    --agent-config-path '${CONFIG}' --result-path '${result}' \
    --evidence-dir '${EVIDENCE_DIR}' --evidence-jsonl '${evidence_jsonl}' \
    --max-samples '${max_samples}' --n '${n}' --max-turns '${max_turns}' \
    --prompt-length '${prompt_length}' --response-length '${response_length}' \
    --temperature '${temperature}' --top-p '${top_p}' --top-k '${top_k}' \
    --num-workers 1 --n-gpus-per-node 2 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 ${specrl_flags}; ray stop --force || true" 2>&1 | tee -a "${LOG}"
  log "finish run=${run_name} result=${result} evidence=${evidence_jsonl}"
}

stop_cache_server() {
  local pids
  pids="$(docker exec uniagent-specrl pgrep -f '[s]tart_specrl_cache_server.py' || true)"
  if [[ -n "${pids}" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] && docker exec uniagent-specrl kill "${pid}"
    done <<< "${pids}"
  fi
}

start_clean_cache_server() {
  stop_cache_server
  docker exec uniagent-specrl rm -rf /dev/shm/SUFFIX_CACHE
  docker exec uniagent-specrl bash -lc "${SPECRL_ENV}; nohup python3 -u examples/agent_interaction/start_specrl_cache_server.py > '${OUT}/swe_lego_qwen3_8b_cache_${STAMP}.log' 2>&1 &"
  sleep 5
  docker exec uniagent-specrl bash -lc "cd ${REPO}; python3 examples/agent_interaction/check_specrl_ready.py; python3 -c 'import socket; s=socket.create_connection((\"127.0.0.1\",6378),timeout=5); s.close(); print(\"cache_server_tcp_ready=127.0.0.1:6378\")'" | tee -a "${LOG}"
}

native_args=(12 8192 4096 0.6 0.95 20)
formal_args=(20 8192 4096 0.6 0.95 20)

case "${MODE}" in
  smoke_vanilla)
    run_one uniagent-eval "swe_lego_qwen3_8b_smoke_vanilla_${STAMP}" 2 1 "${native_args[@]}" ""
    ;;
  smoke_specrl)
    start_clean_cache_server
    run_one uniagent-specrl "swe_lego_qwen3_8b_smoke_specrl_warm_${STAMP}" 2 1 "${native_args[@]}" "--enable-specrl --specrl-update-cache"
    run_one uniagent-specrl "swe_lego_qwen3_8b_smoke_specrl_cached_round1_${STAMP}" 2 1 "${native_args[@]}" "--enable-specrl"
    ;;
  quality)
    run_one uniagent-eval "swe_lego_qwen3_8b_quality_vanilla_16_n2_${STAMP}" 16 2 "${formal_args[@]}" ""
    ;;
  formal)
    quality_result="$(ls -1t "${OUT_HOST}"/swe_lego_qwen3_8b_quality_vanilla_16_n2_*.json 2>/dev/null | head -1 || true)"
    [[ -n "${quality_result}" ]] || { echo "No quality result; run '$0 quality' first" >&2; exit 3; }
    python3 - "${quality_result}" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
effective=d.get('resolved',0)>0 or any(not x.get('patch_empty',True) for x in d.get('instances',[]))
if not effective:
    raise SystemExit('Quality gate failed: no resolved rollout or non-empty patch; diagnose evidence before formal')
print('quality_gate_passed',sys.argv[1])
PY
    vanilla="swe_lego_qwen3_8b_formal_vanilla_${STAMP}"
    warm="swe_lego_qwen3_8b_formal_specrl_warm_${STAMP}"
    c1="swe_lego_qwen3_8b_formal_specrl_cached_round1_${STAMP}"
    c2="swe_lego_qwen3_8b_formal_specrl_cached_round2_${STAMP}"
    c3="swe_lego_qwen3_8b_formal_specrl_cached_round3_${STAMP}"
    run_one uniagent-eval "${vanilla}" 64 2 "${formal_args[@]}" ""
    start_clean_cache_server
    run_one uniagent-specrl "${warm}" 64 2 "${formal_args[@]}" "--enable-specrl --specrl-update-cache"
    run_one uniagent-specrl "${c1}" 64 2 "${formal_args[@]}" "--enable-specrl"
    run_one uniagent-specrl "${c2}" 64 2 "${formal_args[@]}" "--enable-specrl"
    run_one uniagent-specrl "${c3}" 64 2 "${formal_args[@]}" "--enable-specrl"
    docker exec uniagent-specrl bash -lc "cd ${REPO}; python3 examples/agent_interaction/summarize_swe_lego_8b.py \
      --result '${OUT}/${vanilla}.json' --result '${OUT}/${warm}.json' \
      --result '${OUT}/${c1}.json' --result '${OUT}/${c2}.json' --result '${OUT}/${c3}.json' \
      --output '${OUT}/swe_lego_qwen3_8b_formal_summary_${STAMP}.json'" | tee -a "${LOG}"
    ;;
  control_swe_lego)
    run_one uniagent-eval "swe_lego_qwen3_8b_control_oldparams_vanilla_16_n2_${STAMP}" 16 2 8 4096 2048 0.8 0.9 -1 ""
    ;;
  *)
    echo "usage: $0 {smoke_vanilla|smoke_specrl|quality|formal|control_swe_lego}" >&2
    exit 2
    ;;
esac

log "mode_done=${MODE} stamp=${STAMP}"
