#!/bin/bash
# 复现性验证 v3：kvc×3 + sticky×3，每 run timeout 45min 容错，崩/超时不阻塞后续。
# 前置: vllm-src worker.py:342 token_ids 补丁须已打(见 mooncake-tokenids-none-fix),否则 kvc 必崩 TypeError。
# 每 kvc run 前重启单实例 mooncake daemon(卫生, 防多实例叠加)。
# 🔑 必须在容器内执行: docker exec -d hgq-swe-vllm021 bash /data1/hgq/repro_runs.sh
#    (宿主无 mooncake/verl/vllm 二进制; infer_multi.sh 直接 python parallel_infer.py)
cd /data1/hgq/uni-agent
SUM=/tmp/repro_summary.csv
echo "run,strategy,gini,maxmin,gen_thru_mean,wallclock_min,rm_score,status" > $SUM
echo "[start $(date +%H:%M:%S)] 6 runs begin (v3, per-run mooncake restart, 45min timeout each)" >> /tmp/repro_watcher.log

# 重启单实例 mooncake daemon (容器内执行, mooncake 在 PATH)。kvc run 前必调。
restart_mooncake() {
  echo "[mooncake $(date +%H:%M:%S)] restart single instance" >> /tmp/repro_watcher.log
  pkill -9 -f mooncake 2>/dev/null
  for i in $(seq 1 15); do  # 等端口释放
    ss -ltn 2>/dev/null | grep -qE ":9422|:9527" || break
    sleep 1
  done
  mooncake_http_metadata_server --port 9527 --host 127.0.0.1 > /tmp/mooncake_meta.log 2>&1 &
  sleep 3
  MOONCAKE_CONFIG_PATH=/data1/hgq/mooncake_config.json mooncake_master --rpc_port 9422 > /tmp/mooncake_master.log 2>&1 &
  sleep 5
  local N=$(ss -ltn 2>/dev/null | grep -cE ":9422|:9527")
  if [ "$N" != "2" ]; then echo "[mooncake WARN] listening ports=$N (expect 2=meta+master)" >> /tmp/repro_watcher.log; fi
}

kill_mooncake() {  # sticky run 前清掉(sticky 不用 mooncake, 释放资源 + 公平对比)
  pkill -9 -f mooncake 2>/dev/null
}

run_one() {
  local name=$1 strategy=$2
  local LOG=/tmp/repro_${name}.log
  echo "[run $name $(date +%H:%M:%S)] start strategy=$strategy" >> /tmp/repro_watcher.log
  docker rm -f $(docker ps -aq --filter name=uni-agent-) >/dev/null 2>&1
  sleep 8  # 等 ZMQ 端口释放
  if [ "$strategy" = "kvc" ]; then
    restart_mooncake  # 🔑 每 kvc run 单实例 daemon, 防多实例污染
  else
    kill_mooncake
  fi
  local START=$(date +%s)
  if [ "$strategy" = "kvc" ]; then
    timeout 2700 env ENABLE_MOONCAKE=1 MOONCAKE_CONFIG_PATH=/data1/hgq/mooncake_config.json \
      ROUTER_CONFIG=pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml \
      PYTHONPATH=/data1/hgq/uni-agent:/data1/hgq/uni-agent/verl VLLM_HOST_IP=127.0.0.1 \
      RAY_DEDUP_LOGS=0 NCCL_DEBUG=INFO \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=1 NGPUS=8 NWORKERS=8 \
      MAX_SAMPLES=16 N=4 MAX_NUM_SEQS=8 PROMPT_LEN=16384 RESPONSE_LEN=8192 \
      bash scripts/infer_multi.sh /data1/models/Qwen/Qwen3-8B scripts/swe_bench_verified_modal.parquet > $LOG 2>&1
    local RC=$?
  else
    timeout 2700 env PYTHONPATH=/data1/hgq/uni-agent:/data1/hgq/uni-agent/verl \
      RAY_DEDUP_LOGS=0 NCCL_DEBUG=INFO \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=1 NGPUS=8 NWORKERS=8 \
      MAX_SAMPLES=16 N=4 MAX_NUM_SEQS=8 PROMPT_LEN=16384 RESPONSE_LEN=8192 \
      bash scripts/infer_multi.sh /data1/models/Qwen/Qwen3-8B scripts/swe_bench_verified_modal.parquet > $LOG 2>&1
    local RC=$?
  fi
  # 清残留 vllm/ray，防下个 run ZMQ 冲突
  pkill -9 -f parallel_infer 2>/dev/null; pkill -9 -f VLLM 2>/dev/null
  pkill -9 -f EngineCore 2>/dev/null; pkill -9 -f gcs_server 2>/dev/null
  pkill -9 -f raylet 2>/dev/null; pkill -9 -f "ray::" 2>/dev/null
  local ENGCORE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr "\n" " ")
  [ -n "$ENGCORE" ] && kill -9 $ENGCORE 2>/dev/null
  sleep 5
  local END=$(date +%s); local MINS=$(( (END-START)/60 ))
  local STATUS="ok"; [ "$RC" -ne 0 ] && STATUS="timeout_or_err(rc=$RC)"
  local GINI="NA" MAXMIN="NA"
  if [ "$strategy" = "kvc" ]; then
    GINI=$(bash /data1/hgq/analyze_load_imbalance.sh $LOG 2>/dev/null | grep -oE "gini=[0-9.]+" | head -1 | sed s/gini=//)
    MAXMIN=$(bash /data1/hgq/analyze_load_imbalance.sh $LOG 2>/dev/null | grep -oE "max/min = [0-9.]+x" | head -1 | grep -oE "[0-9.]+")
  else
    GINI=$(bash /data1/hgq/analyze_sticky_gini.sh $LOG 2>/dev/null | grep -oE "gini=[0-9.]+" | head -1 | sed s/gini=//)
    MAXMIN=$(bash /data1/hgq/analyze_sticky_gini.sh $LOG 2>/dev/null | grep -oE "max/min=[0-9.]+x" | head -1 | grep -oE "[0-9.]+")
  fi
  local THRU=$(grep "Avg generation throughput" $LOG | grep -oE "Avg generation throughput: [0-9.]+" | grep -oE "[0-9.]+$" | awk "\$1+0>0{s+=\$1;n++} END{if(n>0)printf \"%.1f\",s/n; else print \"NA\"}")
  local RM=$(grep "Mean RM Score" $LOG | tail -1 | grep -oE "[0-9.]+")
  # 崩溃检测: 记录 EngineDead/TypeError 计数便于诊断
  local ED=$(grep -c EngineDeadError $LOG 2>/dev/null)
  local TY=$(grep -cE "NoneType.*not iterable|object is not iterable" $LOG 2>/dev/null)
  [ "$ED" -ge 1 ] && STATUS="$STATUS EngineDead=$ED"
  [ "$TY" -ge 1 ] && STATUS="$STATUS TypeError=$TY"
  echo "$name,$strategy,$GINI,$MAXMIN,$THRU,$MINS,$RM,$STATUS" >> $SUM
  echo "[run $name done $(date +%H:%M:%S)] rc=$RC gini=$GINI thru=$THRU ${MINS}min ED=$ED TY=$TY status=$STATUS" >> /tmp/repro_watcher.log
}

run_one K1 kvc
run_one K2 kvc
run_one K3 kvc
run_one S1 sticky
run_one S2 sticky
run_one S3 sticky
echo "[all done $(date +%H:%M:%S)]" >> /tmp/repro_watcher.log
cat $SUM
