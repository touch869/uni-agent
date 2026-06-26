#!/bin/bash
# 复现性结论分析：读 /tmp/repro_summary.csv，算 kvc/sticky 各指标的均值±std，判断稳定性
# 用法: docker exec hgq-swe-vllm021 bash /data1/hgq/analyze_repro_summary.sh
CSV=/tmp/repro_summary.csv
if [ ! -f "$CSV" ]; then echo "[summary 不存在]"; exit 1; fi

echo "========== 复现性验证结论 =========="
echo "原始数据:"; cat "$CSV"; echo

awk -F, 'NR>1 {
  gini[$2]=gini[$2]" "$3; thru[$2]=thru[$2]" "$5; wc[$2]=wc[$2]" "$6
  ngini[$2]++; nthru[$2]++; nwc[$2]++
}
END{
  for(s in ngini){
    # 基尼
    n=0; sg=0;
    split(gini[s],ga," ");
    for(i in ga){ if(ga[i]!=""){v=ga[i]+0; g[n++]=v; sg+=v; if(n==1){mn=v;mx=v} if(v<mn)mn=v; if(v>mx)mx=v } }
    mg=sg/n; sd=0; for(i=0;i<n;i++){d=g[i]-mg; sd+=d*d} sd=sqrt(sd/n); cv=(mg>0)?sd/mg:0;
    # throughput
    nt=0; st=0; split(thru[s],ta," ");
    for(i in ta){ if(ta[i]!=""){v=ta[i]+0; t[nt++]=v; st+=v } }
    mt=st/nt; sdt=0; for(i=0;i<nt;i++){d=t[i]-mt; sdt+=d*d} sdt=sqrt(sdt/nt);
    # wallclock
    nw=0; sw=0; split(wc[s],wa," ");
    for(i in wa){ if(wa[i]!=""){v=wa[i]+0; w[nw++]=v; sw+=v } }
    mw=sw/nw; sdw=0; for(i=0;i<nw;i++){d=w[i]-mw; sdw+=d*d} sdw=sqrt(sdw/nw);
    printf "%s: 基尼 均值=%.3f std=%.3f CV=%.2f 范围[%.3f,%.3f] | gen_thru 均值=%.1f±%.1f | wall-clock 均值=%.0f±%.0fmin\n", s, mg, sd, cv, mn, mx, mt, sdt, mw, sdw;
  }
}' "$CSV"

echo
echo "========== 复现标准判定 =========="
# kvc 基尼稳定性 + kvc vs sticky wall-clock
awk -F, 'NR>1 {gini[$2]=gini[$2]" "$3; wc[$2]=wc[$2]" "$6; ng[$2]++}
END{
  # kvc 基尼 CV
  s="kvc"; n=0; sg=0; split(gini[s],ga," ");
  for(i in ga){if(ga[i]!=""){v=ga[i]+0; g[n++]=v; sg+=v}} mg=sg/n; sd=0;
  for(i=0;i<n;i++){d=g[i]-mg; sd+=d*d} sd=sqrt(sd/n); cvk=(mg>0)?sd/mg:0;
  printf "①基尼稳定(kvc CV): %.3f %s\n", cvk, (cvk<0.30?"✓ 稳定(<0.3)":"✗ 不稳定(>=0.3)");
  # sticky 基尼 CV
  s="sticky"; n=0; sg=0; split(gini[s],ga," ");
  for(i in ga){if(ga[i]!=""){v=ga[i]+0; g[n++]=v; sg+=v}} mg=sg/n; sd=0;
  for(i=0;i<n;i++){d=g[i]-mg; sd+=d*d} sd=sqrt(sd/n); cvs=(mg>0)?sd/mg:0;
  printf "①基尼稳定(sticky CV): %.3f %s\n", cvs, (cvs<0.30?"✓ 稳定":"✗ 不稳定");
  # wall-clock: kvc 均值 vs sticky 均值
  s="kvc"; n=0; sw=0; split(wc[s],wa," "); for(i in wa){if(wa[i]!=""){sw+=wa[i]; n++}} mk=sw/n;
  s="sticky"; n=0; sw=0; split(wc[s],wa," "); for(i in wa){if(wa[i]!=""){sw+=wa[i]; n++}} ms=sw/n;
  printf "②吞吐对比 wall-clock: kvc均值=%.0fmin vs sticky均值=%.0fmin %s\n", mk, ms, (mk>ms?"✓ kvc稳定更慢(吞吐更低)":"✗ kvc不比sticky慢");
}' "$CSV"
