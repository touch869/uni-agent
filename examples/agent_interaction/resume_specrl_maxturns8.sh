#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data1/zpy/workspace/output/swebench_eval"
RUN_TAG="maxturns8_64_n2_gpu060"
LOG_PATH="${OUT_DIR}/${RUN_TAG}_resume_specrl.log"
mkdir -p "${OUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG_PATH}"
}

run_in_container() {
  local container="$1"
  local command="$2"
  log "start ${container}: ${command}"
  docker exec "${container}" bash -lc "${command}" 2>&1 | tee -a "${LOG_PATH}"
  log "finish ${container}"
}

BASE_ARG_STR="--data-path /workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet --model-path /workspace/models/Qwen3-1.7B-base --agent-config-path /workspace/uni-agent/examples/agent_interaction/agent_config_local.yaml --engine vllm --max-samples 64 --max-turns 8 --prompt-length 4096 --response-length 2048 --temperature 0.8 --top-p 0.9 --n 2 --num-workers 1 --n-gpus-per-node 1 --tensor-parallel-size 1 --gpu-memory-utilization 0.60"
SPECRL_ENV="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1; cd /workspace/uni-agent; export VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0"

log "resume_begin ${RUN_TAG}"

run_in_container "uniagent-specrl" "${SPECRL_ENV}; ray stop --force || true; rm -rf /dev/shm/SUFFIX_CACHE; nohup python3 -u examples/agent_interaction/start_specrl_cache_server.py > /workspace/output/swebench_eval/specrl_cache_${RUN_TAG}.log 2>&1 & sleep 5; python3 examples/agent_interaction/parallel_infer.py ${BASE_ARG_STR} --enable-specrl --specrl-update-cache --result-path /workspace/output/swebench_eval/specrl_warm_${RUN_TAG}.json; ray stop --force || true"

for round in 1 2 3; do
  run_in_container "uniagent-specrl" "${SPECRL_ENV}; ray stop --force || true; python3 examples/agent_interaction/parallel_infer.py ${BASE_ARG_STR} --enable-specrl --result-path /workspace/output/swebench_eval/specrl_cached_${RUN_TAG}_round${round}.json; ray stop --force || true"
done

run_in_container "uniagent-specrl" "python3 - <<'PY2'
import json
paths=[
('/workspace/output/swebench_eval/vanilla_maxturns8_64_n2_gpu060.json', 'vanilla'),
('/workspace/output/swebench_eval/specrl_warm_maxturns8_64_n2_gpu060.json', 'specrl_warm'),
('/workspace/output/swebench_eval/specrl_cached_maxturns8_64_n2_gpu060_round1.json', 'specrl_cached_round1'),
('/workspace/output/swebench_eval/specrl_cached_maxturns8_64_n2_gpu060_round2.json', 'specrl_cached_round2'),
('/workspace/output/swebench_eval/specrl_cached_maxturns8_64_n2_gpu060_round3.json', 'specrl_cached_round3'),
]
base_ids=base_wall=base_gen=None
rows=[]
for i,(path,name) in enumerate(paths):
    d=json.load(open(path))
    ids=[s.get('instance_id') for s in d.get('samples', [])]
    scores=d.get('rm_scores', [])
    if i == 0:
        base_ids=ids
        base_wall=d.get('wall_time_s')
        base_gen=d.get('generation_time_s')
    wall=d.get('wall_time_s')
    gen=d.get('generation_time_s')
    rows.append({
        'run': name,
        'num_samples': d.get('num_samples'),
        'unique_instances': len(set(ids)),
        'n': d.get('n'),
        'max_turns': d.get('max_turns'),
        'mean_rm_score': d.get('mean_rm_score'),
        'resolved': sum(1 for s in scores if s > 0),
        'wall_time_s': wall,
        'generation_time_s': gen,
        'wall_speedup': base_wall / wall if wall else None,
        'gen_speedup': base_gen / gen if gen else None,
        'same_ids_as_vanilla': ids == base_ids,
    })
out='/workspace/output/swebench_eval/summary_maxturns8_64_n2_gpu060.json'
with open(out, 'w') as f:
    json.dump(rows, f, indent=2)
print(json.dumps(rows, indent=2))
print('summary_path=' + out)
PY2"

log "resume_done ${RUN_TAG}"
