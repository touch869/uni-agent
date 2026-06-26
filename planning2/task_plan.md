# Task Plan: Mooncake TP=2 充分验证(150 hgq-swe)

## Goal
在 root@8.92.9.150 容器 hgq-swe 上,**充分验证** TP=2 + mooncake 跨 vllm KV 传递在持续负载下稳定。成功标准:5 轮多轮 + 20 并发 + 3 分钟持续负载下,**TRANSFER_FAIL 全程 = 0**。

## 关键原则(重要)
- **默认用原始路径**(MOONCAKE_CPU_STAGING=0)。只有当原始路径出现 TRANSFER_FAIL 时才启用 CPU staging 安全网重验。
- 之前 subagent 的验证用了 CPU staging=1(已开安全网),所以"无法复现 TRANSFER_FAIL"的结论**不能直接采信**——必须用原始路径重验。
- 充分验证 = 多轮(5 轮不同 prompt)+ 高并发(20 同时)+ 持续负载(3 分钟),逐步加重。

## 环境(已勘探)
- 容器:`hgq-swe`(root@8.92.9.150)
- GPU:8 × RTX 3090 24GB(0,1 → A;2,3 → B)
- 模型:`/data1/models/Qwen/Qwen3-8B`(首选)或 `/data1/lll/workspace/models/Qwen2.5`
- vllm:0.21.0(editable @ /data1/hgq/vllm-src)
- mooncake:transfer-engine 0.3.11.post1
- mooncake worker 补丁:已装 MOONCAKE_CPU_STAGING env(default 0)
- CUDA compat:OK(无需 mv)
- 端口:9422(master) / 9527(metadata) / 8001(A) / 8002(B)

## 之前的 subagent 发现(参考)
- 上次跑用了 `MOONCAKE_CPU_STAGING=1`(已开安全网),TRANSFER_FAIL=0,External hit=99.8%
- **但**:用 CPU staging=1 的结果不能证明原始路径(CUDA memory copy)稳定
- master metrics 全 0(PutStart=0/Get=0),但 meta server 有 PUT/GET 流量(2927B payload)——mooncake 架构:metadata server 做 peer discovery,master 记 KV 传输(可能 metric 节点选错)

## Phases

### Phase 0: 环境清理 + 脚本准备(🔄 in_progress)
- [x] 勘探 150 环境(GPU/模型/mooncake/vllm 版本)
- [x] 清理 stale 进程(master/metadata/vllm 全 kill,端口已 free,GPU 已 free)
- [ ] 写启动脚本(start_mooncake.sh / start_vllm.sh 原始路径 MOONCAKE_CPU_STAGING=0)
- [ ] 写 bench 脚本(bench.py:5 轮 + 20 并发 + 3min 持续)
- [ ] 写 check 脚本(check.sh:抓 TRANSFER_FAIL / master metrics / External hit)
- **Status:** in_progress

### Phase 1-5: 验证全部执行完(结果见下方)✅ done
- 环境:Qwen3-8B, TP=2 × 2 instance, GPU 0-3, kv_both, max_model_len=8192, 2000-4000 token prompts
- 验证了 3 个配置变体:(a) 原始路径 lease=5s, (b) CPU staging=1 lease=5s, (c) 原始路径 lease=60s
- **结果:全部 3 个变体的高并发/持续负载都 FAIL。round5 在干净状态 PASS,污染状态 FAIL**

### 验证结果汇总(TRANSFER_FAIL 计数,越少越好,目标=0)
| 变体 | round5 干净 | concurrent20 | sustained180(3min) |
|---|---|---|---|
| 原始路径 lease=5s | **0 PASS** | **1009 FAIL** | 未测 |
| CPU staging=1 lease=5s | **0 PASS** | **2235 FAIL** | 未测 |
| 原始路径 lease=60s | 690 FAIL(污染后) | **2507 FAIL** | **3450 FAIL** |

### 失败模式诊断(根因定位)
1. **LEASE_EXPIRED(B 端 BatchGet)**:lease=5s 时,并发下 put 慢 → 5s 内 lease 过期 → B get 失败。**lease 改 60s 后 LEASE_EXPIRED=0**(此子因修复)
2. **code={-800} TRANSFER_FAIL (Transfer 0 failed)**(A 端 batch_put):lease 改 60s 后仍持续。这是 mooncake transport 层(vllm worker 间 peer-to-peer TCP KV 传输)失败,与 lease/CUDA 都无关。batch_bytes=53-194MB(超大 batch)
3. **CPU staging 不能修复 -800**:staging=1 确认生效(无 GPU copy fallback 错误),但 -800 FAIL 数量与原始路径相同 → staging 假设(CUDA memcpy 跨 context 失败)被推翻
4. **master 健康**:AllocFail=0, Eviction Success/Attempts=0/0, Mem Storage 充足 → 不是 master 内存/驱逐问题

### 关键发现(推翻 subagent 之前结论)
- 之前 subagent 报"TP=2 mooncake 无法复现 TRANSFER_FAIL + CPU staging 是安全网" **是错的**:
  - 它只跑了低并发 + CPU staging=1 → 没触发
  - 实测 20 并发长 prompt(2000-4000 token)就能稳定触发 TRANSFER_FAIL(1000-2500 次)
  - CPU staging 不是安全网(-800 是 transport 层失败,staging 改的是 GPU→CPU memcpy 层,两层无关)
- **真根因**:mooncake transfer engine 在大 batch(50-200MB)+ 并发 KV 传输下的 transport 层(TCP)失败,非 CUDA memcpy、非 lease、非 master 内存
- **RTX 3090 PCIe 无 P2P** 不是直接原因(subagent 已验证 cudaMemcpy Default 走 primary context OK);transport 层失败更可能是 batch 聚合/连接/带宽问题

### 下一步建议(不在本任务范围)
- 调小 batch_put 的 batch_bytes(改 worker.py 分片)看 -800 是否消失
- 试 mooncake TCP 之外的 protocol(RDMA?但 3090 PCIe 无 NVLink)
- 或接受 TP=2 + mooncake 不适合高并发长 prompt,KVCAware router 用 TP=1(dp 更多)或 TP=2 但开 sticky session 不跨 replica KV

## 关键设计决策
| 决策 | 选择 | 理由 |
|---|---|---|
| 默认 CPU staging | **0(关)** | 必须先验原始路径(CUDA memory copy),才知 TP=2 是否真稳定 |
| TP | 2 | 任务要求;RTX 3090 PCIe 无 P2P 是潜在风险点 |
| 模型 | Qwen3-8B(首选) | 任务要求;Qwen2.5 兜底 |
| max-model-len | 8192 | 任务要求(长 prompt 用 2000-4000 token 内) |
| 传输配置 | kv_both(both 角色) | 两个 vllm 都既能 put 又能 get,简单 |
| 长 prompt | 2000-4000 token | 任务要求,触发 KV offload(短 prompt GPU 命中不 offload) |

## 成功标准
✅ TP=2 + mooncake 5 轮 + 20 并发 + 3min 持续负载下,**TRANSFER_FAIL 全程 = 0**
- (辅助指标)master batch_put/batch_get 成功率 100%
- (辅助指标)B 的 External prefix cache hit rate > 0%
- (辅助指标)vllm throughput 正常(非 0)
