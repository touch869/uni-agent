# 工作进展文档(持续维护,防遗忘)

## 工作背景
8B/Qwen3-8B 在 swebench-verified 上的 KVCAware router 超参搜索实验。
4 台机器(144/145/146/147) 各 8×RTX 3090 24GB,分发 18 组 KVCAware(alpha × load_threshold)+ 对照(sticky default)。

## 环境

### 机器
- 144/145/146: 容器 `hgq-swe`(ssh root@8.92.9.14x)
- 147: 容器 `hgq-swe-vllm021`(ssh root@8.92.9.147)
- 150: mooncake 验证环境(容器 `hgq-swe`,ssh root@8.92.9.150)
- WSL 本地: /home/hgq/workspace/aicoder/ws-uni-agent/uni-agent

### 代码路径
- 仓库: /data1/hgq/uni-agent(四台) — llm-router-mock 分支(非 git, scp 同步)
- verl: /data1/hgq/uni-agent/verl(vendored, editable install)
- vllm: /data1/hgq/vllm-src(v0.21.0, editable)
- mooncake: mooncake-transfer-engine 0.3.11.post1

### 关键配置文件
- sweep 脚本: /data1/hgq/uni-agent/sweep_capacity_load.sh
- infer 脚本: /data1/hgq/uni-agent/scripts/infer_multi.sh
- agent 配置: /data1/hgq/uni-agent/scripts/agent_config_mock.yaml(type:mock, max_turns:50)
- mooncake config: /data1/hgq/uni-agent/mooncake_config.json
- 结果日志: /data1/hgq/sweep_results/*.log
- traj 日志: /data1/hgq/agentic_log_mock/<run_id>/run.log

## 实验设计

### 控制变量(关键)
| 参数 | 对照(sticky) | KVCAware | 说明 |
|---|---|---|---|
| TP | **2** | **2** | 锁死 |
| MAX_NUM_SEQS | **64** | **64** | 锁死 |
| CONCURRENCY | **128** | **128** | 锁死 |
| MAX_SAMPLES | **32** | **32** | 锁手 |
| PROMPT_LEN | **31744** | **31744** | 锁死 |
| RESPONSE_LEN | **8192** | **8192** | 锁死 |
| MAX_MODEL_LEN | **40960** | **40960** | 锁死 |
| max_turns | **50** | **50** | 锁死 |
| ENABLE_MOONCAKE | **0** | **1** | 对照不开,KVCAware 开(mooncake 是 KVCAware 跨 replica KV 共享的基础设施,不是实验变量本身) |
| ROUTER_CONFIG | "" | pkg://kvc_aware | **唯一实验变量** |

### 说明
- TP=2 锁死(对照和 KVCAware 一致)
- mooncake 状态(见 §4): lease=60s 已修(LEASE_EXPIRED=0); transport -800/writeBody(tp_rank:1)未解, 150 subagent 攻关中。CPU staging 已证无效(非安全网)
- mooncake -800 解决后, KVCAware 搜索用 TP=2 + ENABLE_MOONCAKE=1
- 对照组不开 mooncake(sticky default 不需要跨 replica KV 共享)
- ⚠️ **控制变量 caveat(交接必读)**: 对照组用**原始代码**跑的(已完成); KVCAware 搜索将用**负载修复后代码**(commit eb097c8, §6c)。即对照 vs KVCAware 的代码基线不同——对照不受 router 改动影响(它走 verl 内置 sticky, ROUTER_CONFIG=""),但需在分析时注明 KVCAware 侧含负载修复

### 对照组命令(当前跑中)
```bash
cd /data1/hgq/uni-agent
export PYTHONPATH=/data1/hgq/uni-agent ENABLE_MOONCAKE=0 MAX_MODEL_LEN=40960 \
       CONCURRENCY=128 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAX_SAMPLES=32 N=4 NWORKERS=8 TP=2 MAX_NUM_SEQS=64 \
  PROMPT_LEN=31744 RESPONSE_LEN=8192 ROUTER_CONFIG="" \
  bash scripts/infer_multi.sh /data1/models/Qwen/Qwen3-8B \
  /data1/hgq/uni-agent/scripts/swe_bench_verified_modal.parquet \
  scripts/agent_config_mock.yaml
# = 128 traj/组(32 prompt × N=4)
```

### KVCAware 搜索命令(对照完成后, mooncake 修复后)
```bash
# 同上所有参数, 改两个:
# 1. ENABLE_MOONCAKE=1(mooncake 修复后)
# 2. ROUTER_CONFIG=pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml
# alpha/load_threshold 通过 sed patch kvc_aware_strategy.yaml
# round-robin 分发: HOST_INDEX=0/1/2/3 NUM_HOSTS=4
# alpha=[0.3 0.4 0.5 0.6 0.7 0.8] × load_threshold=[0.9 0.8 0.7] = 18 组
```

### 实验流程(带轨迹备份)
```
每组实验:
1. rm -rf /data1/hgq/agentic_log_mock/* (清旧轨迹)
2. 跑 infer_multi
3. RM Score 出 → cp -r /data1/hgq/agentic_log_mock /data1/hgq/traj_backup/<组名>(备份)
4. 下一组
```

## 问题与解决方案

### 1. 1-token 退化(已解决)
- 根因: vllm_async_server.py:500 max_tokens=min(response_length, prompt_length+response_length-prompt), 单轮假设, 多轮 prompt 累积到 24576 → max_tokens=1
- 修复: 改公式 min(response_length, max_model_len-prompt) + MAX_MODEL_LEN=40960
- commit: 48fa4e3(infer_multi.sh), cf8345b(parallel_infer.py)

### 2. docker overlay ENOSPC(已解决)
- 根因: docker daemon overlay2 graphdriver 状态损坏(非磁盘满)
- 修复: systemctl restart docker

### 3. 146 GPU 驱动(已解决)
- 根因: 146 容器 CUDA compat shim(libcuda.so.575)覆盖宿主(570)→ Error 804
- 修复: mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled

### 4. mooncake TP=2 transport -800 ✅ 已解决(2026-06-26, 三件套: staging + conn-pool + lease)
- **真根因(最终, 三次定位后坐实)**: **TCP ephemeral port 耗尽**。mooncake `TcpTransport` 每条 KV transfer 开新短连接, TP=2×4 worker×高并发 `batch_put` 瞬间打爆宿主 ephemeral port(~28k, 32768-60999)→ `connect: Cannot assign requested address` → `code={-800}`。失败 **tp_rank:0/1 均匀**(非 rank:1), **0 条 writeBody/CUDA**(旧 CUDA 签名已被前次 CPU-staging 补丁消除)。单轮 concurrent20 日志 69844 条 connection 失败
- **修复(纯 env, 不改源码)**: `export MC_TCP_ENABLE_CONNECTION_POOL=1` → TcpTransport 连接池复用, 不再每条 transfer 新 connect, port 耗尽消失
- **150 充分验证(lease=60000)**: concurrent20 TRANSFER_FAIL **1561→0**; sustained180(44 轮)**0 fail, 44/44 过**; External prefix hit **10.9%→99.9%**; B 端 20 并发 **71-104s→12-13s**(KV 命中跳过 prefill)
- **证伪(写进 memory)**: ① batch_put 分片**反更糟**(1561→2199, 更多 submitTransfer=更多短连接=更快 port 耗尽), 已回退; ② VLLM_HOST_IP=127.0.0.1 非必需(conn-pool 一项即 0 fail)
- **⚠️ 更正(2026-06-26 verl 实测)**: 之前判"CPU staging 与本根因无关"**只对 150 standalone**。**verl 路径(4 replica)staging 必需** —— 145 实测 246,445 条 `writeBody failed`(tp_rank:1 CUDA memcpy)。150 因 worker.py 已含 staging 补丁故 writeBody=0; 4 台 verl 未部署 → writeBody 海量。**部署 staging 补丁(scripts/vllm_patches/mooncake_store_worker.py, commit b3b2df6)+ MOONCAKE_CPU_STAGING=1(sweep, cc4c315)后, writeBody 246k→0、TRANSFER_FAIL=0**。即 mooncake TP=2 需**三件套**: staging(writeBody) + conn-pool(端口耗尽) + lease=60s(LEASE_EXPIRED), 缺一不可
- **证据**: 150 `/data1/hgq/mooncake_tp2_verify/`(restart_all.sh 第 40 行含修复 env) + memory `[[mooncake-tp2-conn-pool-fix]]`; 150 已清理(GPU/进程全释放, worker.py 未改, 分片补丁已回退)

### 4-历史. mooncake TP=2 诊断链(早期, 已被上方 conn-pool 结论取代, 保留作 trail)
- **150 充分验证结论(2026-06-26, subagent a19ed91, 2×vllm TP=2 serve + curl 跨 replica 压测)**: TP=2 + mooncake **在 20 并发 + 3min 持续负载下全部失败**, 3 变体 TRANSFER_FAIL 1000-3450 次(目标 0)。vllm fallback 重算故 ok=True, 但 External prefix hit 从 99.6% 跌到 0-13%
- **根因(推翻之前"CUDA memcpy writeBody"结论)**:
  1. **LEASE_EXPIRED**(B 端 BatchGet): mooncake master `default_kv_lease_ttl` 默认 **5000ms(5s)**, 并发下 A put 慢 → lease 在 B get 前过期。**改 `mooncake_master --default_kv_lease_ttl=60000`(60s) → LEASE_EXPIRED=0, 此子因已确认可修复**
  2. **TRANSFER_FAIL code={-800}**(A 端 batch_put): lease 改 60s 后**仍持续**, batch_bytes=53-194MB(超大 batch)。是 mooncake **transport 层(peer-to-peer TCP KV 字节传输)失败**, 与 lease / CUDA memcpy 都无关。**CPU staging(MOONCAKE_CPU_STAGING=1)确认生效但 FAIL 数不变 → 推翻"staging 是安全网"假设**。144 pilot 进一步定位: 失败**全在 `tp_rank:1`**(TP=2 第二块 GPU publish 第二 shard 时 CUDA→网络 memcpy `invalid argument`), 即同 replica 内 TP rank publish 就失败, 非跨 replica 流量问题
- **之前结论更正**: "无法复现 TRANSFER_FAIL + CPU staging 是安全网"**是错的**(那次只跑低并发 + staging=1 没触发)。20 并发长 prompt(2-4k tok)即可稳定触发
- **master 健康**: AllocFail=0, 无 eviction → 非 master 内存/驱逐问题
- **证据/可复现**: 150 `/data1/hgq/mooncake_tp2_verify/{vllm_8001,vllm_8002,master,meta_server}.log` + `bench.py`(3 模式 round5/concurrent20/sustained180)/`restart_all.sh [lease_ms] [cpu_staging]`/`start_vllm.sh`/`start_mooncake.sh`
- **150 已清理**: vllm/mooncake 进程全 kill, GPU free, worker.py debug 日志已删(MOONCAKE_CPU_STAGING env 补丁保留)
- **正在尝试的修复(150 subagent a8e1d66e 后台)**: 改 worker.py 把 batch_put 分片(batch_bytes<10MB)+ 查 tp_rank:1 CUDA context(transport memcpy 前 cudaSetDevice), 看 -800 是否消失
- **注意**: subagent 压测是"每次请求强制跨 replica"的极限场景; verl KVCAware 实战以 sticky-session 为主(偶尔跨 replica), 实战 TRANSFER_FAIL 可能远低于压测

### 5. 无 mooncake KVCAware 探针(插曲实验, 145/146/147 空闲利用)
- **目的**: 利用对照完成后空闲的 145/146/147, 测试 KVCAware router 不开 mooncake 的耗时(预估 KVCAware 全搜索时间)
- **配置**: 同对照参数(TP=2, MAX_NUM_SEQS=64, CONCURRENCY=128), 只改 ROUTER_CONFIG=pkg://kvc_aware
- **3 台测试不同 (alpha, threshold)**:
  - 147: alpha=0.5, threshold=0.9
  - 145: alpha=0.5, threshold=0.7
  - 146: alpha=0.8, threshold=0.9
- **意义**: 这是"KVCAware router 但无跨 replica KV 共享"的数据点, 与对照(sticky default)对比可看纯 router 策略效果

### 6. 无 mc 探针极慢的根因(2026-06-26 诊断, 重要)—— 冷启动平局 + sticky 锁死
- **现象**: 对照(sticky)~15min 跑完 128 traj; 无 mc 探针 ~18min 只完成 9-17/128, 预计 ~4h/台。GPU 0,1=100% 但 **2-7 持续 0%**(8 卡只用 2 卡)
- **真根因 = 冷启动平局 + sticky 锁死(实锤, 推翻早期"prefix 局部性/采集坏"假说)**:
  - 冷启动: 4 replica 全 idle(kv=0)→ combined score 全 = 0.5。因 α=0.5 使 `S_cache+S_load=1`: idle-cold = 0.5·0+0.5·1.0=0.5; loaded-warm = 0.5·0.7+0.5·0.3=0.5 → **四台完美平局 → ranking[0] 取 dict 第一个 = 22575 → 全部 128 session 初始都绑到 22575**
  - sticky 锁死: 之后 sticky 短路维持; `is_overloaded` 永远 False(load 公式 kv 权重 0.4, kv=1.0 也只 load≤0.4+0.3+0.3, 实测 load=0.70 < threshold 0.9)→ 永不 rebalance
  - 结果: 另 3 replica 从未收到流量 → 恒 idle(kv=0 是**真实值**)、GPU 2-7 = 0%; 22575 KV 撑爆 → prefix 被挤掉 → 命中 30% → 慢
  - **metric 采集其实是好的**: 日志实测 4 replica 各 `polling succeeded` 1040 次、"store refreshed with 4 replica results"。3 replica 显示 kv=0 因**真空闲**(没被路由到), 非采集失败 → 采集团无需修
- **Prefix hit 99%→30% / KV 95-99% 满 / Running~40 Waiting~80 是上述不均的"症状", 非因**(早期诊断顺序写反, 已更正)
- gen throughput ~150-220 tok/s(单 replica ~40 running 聚合)

### 6b. verl 默认调度对比 + s_load 评估(2026-06-26 调研, 用户拍板方向)
- **verl 默认 `GlobalRequestLoadBalancer`(global_sticky_inflight, verl/verl/workers/rollout/router.py:100-154)**: 本地 inflight 计数 `{sid:0}`(acquire++/release--), 新请求 → `min(inflight)`(line 144)。**inflight 本地记账、每决策即时更新 → 冷启动也能均匀分散**(给 A 后 A.inflight=1, 下个新请求 min 选别的)。sticky 只固化, 均衡靠 inflight
- **KVCAware 对比**: 把本地 inflight 换成轮询 kv/running/waiting 打分。轮询虽成功, 但 (a) 滞后(5s 间隔, 非每决策更新)(b) score 公式冷启动平局 → 无 spreading → 钉死单 replica
- **s_load 公式评估(用户结论)**: `load=0.4·kv+0.3·running+0.3·waiting` 本质 ≈ min-inflight(running/waiting)+min-kv, **逻辑合理, 不改**。问题不在公式, 在 **①冷启动 spreading 缺失 ②overload 阈值触发不了**(kv 权重 0.4 使 kv 满 load 也只 ~0.7 < 0.9)
- **修复方向(原调研结论)**: 冷启动 spreading + overload 判定让 kv 能触发。**实际修复见 §6c** —— 最终只改了 waiting_usage 绝对计数 + polling 提速(让 load 能过阈值即可), **冷启动 spreading / inflight 计数并不需要**(验证已达成 8/8 GPU)。metric 采集确认 OK 不修

### 6c. 负载不均修复 —— 已实现 + 验证通过 + pushed(2026-06-26)
**commit eb097c8 (llm-router-mock), 三处改动合力解决, 无需冷启动 spreading/inflight 计数:**
1. `load_score.py`: `waiting_usage` 从**比值** `waiting/(waiting+running+1)` 改**绝对** `min(1, waiting/max_num_seqs)`(和 running 一致)→ replica 跑满(kv满+running=max+backlog)时 load 能达 ~0.9 **过阈值** → `is_overloaded()` 触发 → sticky 释放 → rebalance
2. `collector.py`: `polling_interval` 默认 5→1s + `ROUTER_POLLING_INTERVAL` env 可调(免改代码调频)
3. `configs/collector.yaml`: `polling_interval: 5→1`(YAML 会覆盖 Python 默认, 必须改 YAML 才生效)
4. `collectors/collector/polling_collector.py`: httpx `AsyncClient(trust_env=False)`(commit f678f19)——**144/145/146 的 hgq-swe 容器设了 `HTTP_PROXY=http://8.92.10.60:7890` 且 NO_PROXY 不含 172.17, 导致 poll `http://172.17.x/metrics` 被代理劫持(~1100 失败/台)→ router 盲 → 又钉死 1 replica(2/8 GPU)**。trust_env=False 绕过代理直连 172.17, polling 失败 1100→0, 4 台 spread 恢复(OVERLOADED 打火, 2/8→4-8/8)。**147 不受影响**(其 replica 用 host IP 8.92.9.147)。参考 [[kvc-router-proxy]]

**验证(147 hgq-swe-vllm021, alpha=0.5/load_threshold=0.7, no-mc, polling=1):**
| 指标 | 修复前(钉死1台) | 修复后 |
|---|---|---|
| GPU 活跃 | 2/8 | **8/8 全 100%** |
| gen 吞吐 | ~15 tok/s | **1325 tok/s**(88×) |
| Prefix hit | ~14% | **81.8%** |
| 路由分布 | 全给 1 台 | 4 replica 均衡(29/23/5/3) |
| OVERLOADED 触发 | 0(永不) | 40+(rebalance 活跃) |
| combined scores | 全 0.5 平局 | 有区分(0.57/0.55/0.49/0.45) |

→ **负载不均已解决**。`load_threshold=0.7`(降阈值→更早 overload→更快分散)效果最好。
- **对照无需重跑**: control 用 verl 内置 sticky(`ROUTER_CONFIG=""`), 不经 KVCAware router, 本次改动零影响, 对照基准(RM=0.0000、~790-1080s、prefix 99%)有效
- **待同步**: 144/145/146 需 scp 这 3 文件(147 已部署验证)

### 7. ~~mooncake 未装~~ → 更正: mooncake 已装在容器内(2026-06-26 更正, 之前判断错误)
- **更正**: 之前查的是**宿主机** python(`/usr/bin/python3.12` 无 mooncake), 但**实验跑在容器内**(144/145/146=`hgq-swe`, 147=`hgq-swe-vllm021)。容器里 `/usr/local/lib/python3.12/dist-packages/mooncake` + `/usr/local/bin/mooncake_master` 都有, `import mooncake` 正常, vllm 0.21.0
- **实证**: 144 pilot 用 ENABLE_MOONCAKE=1 跑起来, MooncakeStoreConnector 4/4 init 成功, 产生 47658 次 TRANSFER_FAIL → 证明 import + 运行都 OK, 是 **transport 层失败非 import 问题**
- **坑(交接必读)**: `ssh root@8.92.9.14x` 落在**宿主**(hostname ubuntu14x), 实验在**容器**, 操作实验必须 `docker exec hgq-swe`(或 hgq-swe-vllm021)。宿主 python 无 mooncake/pip, 别在宿主上查/装
- **结论**: mooncake **无需安装**, 4 台容器都已具备。阻塞主搜索的是 transport -800(见 §4), 不是安装

## 当前进展(2026-06-26 更新)

### 三大阻塞全清 ✅(本会话 6 个 commit 到 llm-router-mock)
1. **负载不均**(waiting_usage 绝对计数 + polling 1s + threshold, `eb097c8`)→ 见 §6c
2. **mooncake -800**(TCP ephemeral port 耗尽 → `MC_TCP_ENABLE_CONNECTION_POOL=1`, `f19133a`, 150 验证 External hit 99.9%)→ 见 §4
3. **polling 代理劫持**(httpx `trust_env=False`, `f678f19`; 144/145/146 HTTP_PROXY 劫持 172.17 → router 盲)

### 对照组(原始)— 4/4 完成, 基准有效
RM=0.0000, ~790-1080s, prefix 99%。走 verl 内置 sticky, 不经 router, 不受三大修复影响。

### control2 重跑(同相干净基准)🔄 4 台并行
- 用户要求 mc 搜索前重跑一份干净对照(ENABLE_MOONCAKE=0, ROUTER_CONFIG=""), 完成自动备份 `traj_backup/control2_<host>`

### 无 mc KVCAware 搜索(已跑验证 → 已砍)
- 实证 no-mc 慢 = KVCAware overload-release 移动 session, 无 mc 共享 → 重 prefill(见 §6 分析)。用户决策砍掉, 转 mc 优先

### mc KVCAware 主搜索(约束#1 正式搜索)⏭ cron 9e4959d1 自动衔接
- `ENABLE_MOONCAKE=1`, `LOAD_THRESHOLDS="0.7 0.8 0.9"`(**低 threshold 在前**), round-robin 5/5/4/4, `BACKUP_PREFIX=mc`, 每组备份
- **关键验证**: External hit **>0**(mooncake 跨 replica KV 生效) + TRANSFER_FAIL **≈0**(conn-pool) + 吞吐接近对照(mooncake 消除重 prefill)
- lease=60s(sweep) + conn-pool(sweep env) + trust_env(polling) 均已部署 4 台

### 机器状态
- 144/145/146/147: control2 跑中 → 完了转 mc 搜索
- 150: mooncake 验证完成, idle

## 下一步
1. control2 完成 → cron 自动起 mc 搜索
2. mc 搜索完成(External hit/吞吐达标) → `scripts/analyze_results.sh` 出三路对比表
3. 汇总分析(对照/control2 vs mc: 吞吐比/RM/Prefix/External/KV, 见下方指标)
4. 找最优 (alpha, threshold): 吞吐比最高 + External>0 的组合

## 分析指标(最终产出)

### 核心指标
| 指标 | 说明 | 数据来源 |
|---|---|---|
| **总耗时** | 每组实验从启动到 RM Score 的墙钟时间 | sweep_dispatch.log 时间戳 或 infer_multi 开始/结束 |
| **吞吐比值** | 该配置总耗时 / 同机对照组总耗时 | 四台各自算（相对比值屏蔽机器性能差异） |
| RM Score | 解题正确率 | infer_multi 输出 "Mean RM Score" |
| Prefix cache hit rate | vllm 内部 prefix 命中率 | vllm loggers "Prefix cache hit rate: X%" |
| External hit rate | mooncake 跨 replica KV 命中率 | vllm loggers "External prefix cache hit rate: X%" |
| KV cache usage | KV 池利用率 | vllm loggers "GPU KV cache usage: X%" |

### 分析维度
1. **alpha/load_threshold 网格扫描**: 每组 (alpha, threshold) 的总耗时 + RM Score + hit rate, 找最优组合
2. **KVCAware vs 对照(sticky) 吞吐比**: 每台机器上 KVCAware 总耗时 / 对照总耗时, 比值 <1 说明 KVCAware 更快
3. **mooncake 贡献**: External hit rate >0 说明跨 replica KV 复用生效; 对比 throughput 提升
4. **退化率**: completion=1 的 traj 占比(应 ~0%)

### 吞吐比值计算方法(屏蔽机器差异)
```
每台机器:
  对照总耗时 = T_control(本机)
  KVCAware 每组总耗时 = T_kvcaware_aX_ltY(本机)
  吞吐比 = T_control / T_kvcaware_aX_ltY
  > 1.0 = KVCAware 比对照快(提升)
  < 1.0 = KVCAware 比对照慢
四台平均 → 该 (alpha, threshold) 的综合吞吐比
```

### 输出格式(最终表格)
```
alpha | threshold | RM Score | 总耗时 | 吞吐比(vs对照) | Prefix hit | External hit | KV usage
0.3   | 0.9       | 0.06     | 1200s  | 1.15            | 49%        | 0%           | 40%
0.3   | 0.8       | ...
...
control(default)  | 0.05     | 1380s  | 1.00 (基准)    | 45%        | N/A          | 41%
```

## mc 搜索结果矩阵（α × load_threshold）
**值 = 对照组耗时 / 该组耗时（同机），>1.0 = mc 比对照快；<1.0 = mc 更慢**
对照基线(同机): 145≈1080s, 146≈888s, 147≈790s。host 分配(NUM_HOSTS=3, 144 down 未跑): 145=HOST0/146=HOST1/147=HOST2。staging硬开+conn-pool+lease，threshold=0.7/0.8(skip 0.9 劣配)。

| load_th\α | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| **0.7** | 145:0.42(G42%·M28%) | 147 运行中 | 146 待跑 | 145 待跑 | 147 待跑 | 146 待跑 |
| **0.8** | 146 运行中 | 145 待跑 | 147 待跑 | 146 待跑 | 145 待跑 | 147 待跑 |

> 格式: `host:吞吐比(G=GPU prefix%, M=mooncake External%)`。control/mc 均无驱逐(preempt=0)；mc 低 hit 来自跨 replica 移动非驱逐。

