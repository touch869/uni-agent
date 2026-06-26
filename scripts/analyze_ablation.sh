#!/bin/bash
# 分析 A vs D 消融实验结果（在 147 容器内运行）
# 用法: docker exec hgq-swe-vllm021 bash /data1/hgq/analyze_ablation.sh
# 日志: A=/tmp/A_full_n4.log, D=/tmp/D_full_n4.log

A_LOG=${A_LOG:-/tmp/A_full_n4.log}
D_LOG=${D_LOG:-/tmp/D_full_n4.log}

summarize() {
  local name=$1 log=$2
  echo "========== $name ($log) =========="
  if [ ! -f "$log" ]; then echo "  [log 不存在]"; echo; return; fi
  echo "  总行数: $(wc -l < "$log")"
  echo "  --- 错误检查 ---"
  echo "  NCCL err:           $(grep -cE 'unhandled system error|No space left' "$log")"
  echo "  Cannot assign addr: $(grep -c 'Cannot assign requested address' "$log")"
  echo "  writeBody CUDA err: $(grep -c 'writeBody failed to copy from CUDA' "$log")"
  echo "  ImportError:        $(grep -cE 'Cannot resolve|MOONCAKE_CONFIG_PATH is not' "$log")"
  echo "  --- ① BlockStored (KV 写入 mooncake) ---"
  echo "  BlockStored:        $(grep -c 'BlockStored' "$log")"
  echo "  --- ② GPU prefix cache hit (本机 KV 命中) ---"
  grep -oE 'Prefix cache hit rate: [0-9.]+%' "$log" | grep -oE '[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "    gpu hit 均值=%.2f%% (n=%d 采样)\n",s/n,n; else print "    [无 gpu hit 数据]"}'
  grep -oE 'Prefix cache hit rate: [0-9.]+%' "$log" | grep -oE '[0-9.]+' | sort -n | tail -1 | awk '{printf "    gpu hit 峰值=%.1f%%\n",$1}'
  echo "  --- ③ External prefix cache hit (mooncake 跨 replica restore) ---"
  grep -oE 'External prefix cache hit rate: [0-9.]+%' "$log" | grep -oE '[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "    external hit 均值=%.2f%% (n=%d 采样)\n",s/n,n; else print "    [无 external hit 数据]"}'
  grep -oE 'External prefix cache hit rate: [0-9.]+%' "$log" | grep -oE '[0-9.]+' | sort -n | tail -1 | awk '{printf "    external hit 峰值=%.1f%%\n",$1}'
  local total=$(grep -oE 'External prefix cache hit rate: [0-9.]+%' "$log" | wc -l)
  local nonzero=$(grep -oE 'External prefix cache hit rate: [0-9.]+%' "$log" | grep -vE '0.0%' | wc -l)
  echo "    external 非零次数=$nonzero / total=$total"
  grep -oE 'External prefix cache hit rate: [0-9.]+%' "$log" | sort | uniq -c | sort -rn | head -5 | sed 's/^/    分布: /'
  echo "  --- ④ throughput (generation tokens/s) ---"
  grep -oE 'Avg generation throughput: [0-9.]+ tokens/s' "$log" | grep -oE '[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "    gen throughput 均值=%.1f tokens/s (n=%d 采样)\n",s/n,n; else print "    [无 throughput 数据]"}'
  grep -oE 'Avg prompt throughput: [0-9.]+ tokens/s' "$log" | grep -oE '[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "    prompt(prefill) throughput 均值=%.1f tokens/s\n",s/n; else print "    [无 prompt throughput]"}'
  echo "  --- 完成状态 ---"
  grep -E 'Mean RM Score' "$log" | tail -1 | sed 's/^/    /'
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" | head -1 | sed 's/^/    /'
  echo
}

summarize "A 组 (sticky)" "$A_LOG"
summarize "D 组 (router+mooncake α=0.7)" "$D_LOG"

echo "========== watcher 接力状态 =========="
if [ -f /tmp/auto_D_watcher.log ]; then
  cat /tmp/auto_D_watcher.log | sed 's/^/  /'
  if grep -q 'launching D' /tmp/auto_D_watcher.log 2>/dev/null; then
    echo "  ✅ watcher 已触发 D 启动"
  else
    echo "  ⏳ watcher 仍在等 A 完成（D 尚未自动起）"
  fi
fi

echo
echo "========== 结论判读 =========="
echo "  External hit: D 应 >0%（峰值 75%），A 必 =0%（无 mooncake 源）"
echo "  throughput:    D 与 A 对照（mooncake restore 省 prefill，D 应 ≥ A）"
echo "  new_tokens:    D 应 < A（mooncake restore 替代部分重算）"
