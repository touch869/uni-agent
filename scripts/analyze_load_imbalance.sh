#!/bin/bash
# 负载不均分析：解析 router log 的 routed to server= 计数
# 用法: bash analyze_load_imbalance.sh /tmp/xxx.log   (参数优先, 兼容 LOG 环境变量)
LOG=${1:-${LOG:-/tmp/D_full_n4.log}}

if [ ! -f "$LOG" ]; then echo "[log 不存在: $LOG]"; exit 1; fi

echo "========== 负载不均分析 ($LOG) =========="
echo "总路由次数: $(grep -c 'routed to server' "$LOG")"
echo
echo "各 replica 路由数 (降序):"
COUNTS=$(grep -oE 'routed to server=[0-9.]+:[0-9]+' "$LOG" | sort | uniq -c | sort -rn | awk '{print $1}')
echo "$COUNTS" | awk 'BEGIN{n=0} {a[n++]=$1; sum+=$1} END{
  # 排序已降序，a[0]=max, a[n-1]=min
  max=a[0]; min=a[n-1];
  mean=sum/n;
  # 方差/CV
  for(i=0;i<n;i++){d=a[i]-mean; var+=d*d;}
  var/=n; sd=sqrt(var); cv=sd/mean;
  # 基尼系数
  # 先升序排
  for(i=0;i<n;i++) asc[i]=a[i];
  # 简单冒泡升序
  for(i=0;i<n;i++) for(j=i+1;j<n;j++) if(asc[j]<asc[i]){t=asc[i];asc[i]=asc[j];asc[j]=t;}
  for(i=0;i<n;i++){ cum += (2*(i+1)-n-1)*asc[i]; }
  gini = cum / (n*sum);
  printf "  replicas=%d  total=%d  max=%d  min=%d\n", n, sum, max, min;
  printf "  max/min = %.1fx\n", max/min;
  printf "  均值=%.1f  标准差=%.1f  CV(变异系数)=%.3f\n", mean, sd, cv;
  printf "  gini=%.3f  (基尼系数, 0=完全均衡, 1=完全不均)\n", gini;
}'
echo
echo "逐 replica:"
grep -oE 'routed to server=[0-9.]+:[0-9]+' "$LOG" | sort | uniq -c | sort -rn | awk '{printf "  %s -> %s\n", $1, $3}'
