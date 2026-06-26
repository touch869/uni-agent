#!/bin/bash
# 算 sticky 组的负载基尼（口径：各 replica 的 generation throughput 均值分布）
# sticky 无 routed-to-server 日志，用 per-pid throughput 均值作负载代理。
LOG=${1:-/tmp/A64_final.log}
if [ ! -f "$LOG" ]; then echo "[log 不存在: $LOG]"; exit 1; fi

grep "generation throughput" "$LOG" | \
sed -nE 's/.*pid=([0-9]+).*Avg generation throughput: ([0-9.]+) tokens.*/\1 \2/p' | \
awk '{s[$1]+=$2; n[$1]++} END{for(p in s) printf "%d %.2f %d\n", p, s[p]/n[p], n[p]}' | \
sort -k2 -rn > /tmp/_thr.dat

echo "=== 各 replica generation throughput 均值 (pid mean samples, 降序) ==="
cat /tmp/_thr.dat
echo
awk '{a[c++]=$2; sum+=$2} END{
  n=c; max=a[0]; min=a[0];
  for(i=1;i<n;i++){if(a[i]>max)max=a[i]; if(a[i]<min)min=a[i]}
  mean=sum/n; for(i=0;i<n;i++){d=a[i]-mean;var+=d*d} var/=n; sd=sqrt(var); cv=sd/mean;
  for(i=0;i<n;i++)asc[i]=a[i];
  for(i=0;i<n;i++)for(j=i+1;j<n;j++)if(asc[j]<asc[i]){t=asc[i];asc[i]=asc[j];asc[j]=t}
  for(i=0;i<n;i++)cum+=(2*(i+1)-n-1)*asc[i];
  gini=cum/(n*sum);
  printf "replicas=%d max=%.2f min=%.2f max/min=%.2fx mean=%.2f CV=%.3f gini=%.3f\n", n, max, min, max/min, mean, cv, gini
}' /tmp/_thr.dat
