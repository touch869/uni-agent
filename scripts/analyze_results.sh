#!/bin/bash
# 分析 KVCAware 搜索结果: 对照(control2) vs mc 搜索, 出三路对比表。
# 从每台的 sweep_results/<tag>.log 提取: RM Score / 总耗时 / avg Prefix hit / avg KV / avg External hit / TRANSFER_FAIL。
# 吞吐比 = T_control / T_<group>(同机, >1 = 该组比对照快)。
#
# Usage: bash scripts/analyze_results.sh [HOSTS]   (默认 "144 145 146 147")
# 容器: 144/145/146=hgq-swe, 147=hgq-swe-vllm021
set -uo pipefail
HOSTS=${1:-"144 145 146 147"}
CONT_144146="hgq-swe"; CONT_147="hgq-swe-vllm021"
LOGDIR="/data1/hgq/sweep_results"

# 提取单个日志的指标: RM / 总耗时(s) / avg prefix% / avg KV% / avg external% / TRANSFER_FAIL
parse_log() {
  local log="$1" host="$2"
  [ -f "$log" ] || { echo "MISSING"; return; }
  # RM Score
  local rm=$(grep -hoE 'Mean RM Score: [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
  rm=${rm:-NA}
  # 总耗时: 日志首/末 INFO 时间戳差(秒)。格式 "06-26 HH:MM:SS,mmm"(vllm UTC) 或 syslog。
  local t1=$(grep -oE '[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$log" 2>/dev/null | head -1)
  local t2=$(grep -oE '[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$log" 2>/dev/null | tail -1)
  local dur="NA"
  if [ -n "$t1" ] && [ -n "$t2" ]; then
    local s1=$(date -d "${t1#* }" +%s 2>/dev/null), s2=$(date -d "${t2#* }" +%s 2>/dev/null)
    [ -n "$s1" ] && [ -n "$s2" ] && dur=$((s2 - s1))
  fi
  # avg prefix / KV / external hit (vllm loggers lines)
  local pref=$(grep -oE 'Prefix cache hit rate: [0-9.]+' "$log" 2>/dev/null | awk '{sum+=$NF;n++} END{if(n)printf "%.1f",sum/n}')
  local kv=$(grep -oE 'GPU KV cache usage: [0-9.]+' "$log" 2>/dev/null | awk '{sum+=$NF;n++} END{if(n)printf "%.1f",sum/n}')
  local ext=$(grep -oE 'External prefix cache hit rate: [0-9.]+' "$log" 2>/dev/null | awk '{sum+=$NF;n++} END{if(n)printf "%.1f",sum/n}')
  local fail=$(grep -cE 'TRANSFER_FAIL|writeBody failed' "$log" 2>/dev/null)
  echo "${rm}|${dur}|${pref:-NA}|${kv:-NA}|${ext:-NA}|${fail}"
}

printf "# KVCAware 搜索结果分析 (%s)\n\n" "$(date '+%Y-%m-%d %H:%M')"
printf "| host | 组 | RM | 总耗时(s) | 吞吐比(vs对照) | Prefix%% | KV%% | External%% | TRANSFER_FAIL |\n"
printf "|------|-----|------|-----------|----------------|---------|------|-----------|---------------|\n"

for h in $HOSTS; do
  C=$CONT_144146; [ "$h" = "147" ] && C=$CONT_147
  # 对照 control2
  cmetas=$(ssh -o ConnectTimeout=8 root@8.92.9.$h "docker exec $C cat $LOGDIR/control2.log 2>/dev/null" 2>/dev/null > /tmp/c2_$h.log; parse_log /tmp/c2_$h.log $h)
  if [ "$cmetas" != "MISSING" ]; then
    IFS='|' read -r crm cdur cpref ckv cext cfail <<< "$cmetas"
    printf "| %s | control2 | %s | %s | 1.00(基准) | %s | %s | %s | %s |\n" "$h" "$crm" "$cdur" "$cpref" "$ckv" "$cext" "$cfail"
    # mc 组
    for taglog in $(ssh -o ConnectTimeout=8 root@8.92.9.$h "docker exec $C ls $LOGDIR/a*.log 2>/dev/null" 2>/dev/null); do
      tag=$(basename "$taglog" .log)
      ssh -o ConnectTimeout=8 root@8.92.9.$h "docker exec $C cat $taglog 2>/dev/null" 2>/dev/null > /tmp/m_$h.log
      mmetas=$(parse_log /tmp/m_$h.log $h)
      [ "$mmetas" = "MISSING" ] && continue
      IFS='|' read -r mrm mdur mpref mkv mext mfail <<< "$mmetas"
      ratio="NA"
      if [ "$cdur" != "NA" ] && [ "$mdur" != "NA" ] && [ "$mdur" -gt 0 ] 2>/dev/null; then
        ratio=$(awk -v c=$cdur -v m=$mdur 'BEGIN{printf "%.2f",c/m}')
      fi
      printf "| %s | mc_%s | %s | %s | %s | %s | %s | %s | %s |\n" "$h" "$tag" "$mrm" "$mdur" "$ratio" "$mpref" "$mkv" "$mext" "$mfail"
    done
  else
    printf "| %s | (control2 未完成/未找到) | - | - | - | - | - | - | - |\n" "$h"
  fi
done

echo ""
echo "## 说明"
echo "- 吞吐比 = 对照耗时 / 该组耗时, **>1 = 该组比对照快**(KVCAware+mc 目标)"
echo "- External%% >0 = mooncake 跨 replica KV 生效; TRANSFER_FAIL ≈0 = conn-pool 修复生效"
echo "- mc 组最优 alpha/threshold: 吞吐比最高且 External>0 的组合"
