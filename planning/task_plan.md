# Task Plan — 阶段四：llm-router + Mooncake 对 agentic rollout prefix 复用的量化实验

> 驱动后续会话。接手先读 `progress.md` 末尾「当前状态/下一步」，再回本文件。
> 阶段一~三（500 条推理，Mean RM 0.178）已 commit/push，本阶段基于稳定主线。

## 1. 动机

agentic rollout 有两个最值钱的 **prefix 复用** 特征，vLLM 默认调度吃不到满：
- **轮间复用**：turn N 复用 turn N-1 的 prompt+response prefix（多轮对话，上下文累积）
- **sample 间复用**：同一 prompt 的 n 个 rollout 共享前缀（RL 多采样）

KV cache 撑满时，这些 prefix block 被 **LRU 驱逐**（丢弃），下次请求 **miss → 重算 prefill**。30B n=4 已实测 prefix cache 97%→45.2%，驱逐在 RL 多采样场景真实且严重。

**目标架构**：`llm-router（KVCAwareBalancer）+ MooncakeStoreConnector`。Mooncake 作为 **GPU KV cache 的 backing store**——block 被驱逐时**卸到 Mooncake 而非丢弃**，下次从 Mooncake restore，避免重算。router 负责把同源请求导向能复用 prefix 的副本。

## 2. 量化目标（用户定义的三件事）

1. **llm-router vs 一般 sticky 调度** → 吞吐量提升多少
2. **GPU 上的 prefix cache 命中率** 是否提升
3. **Mooncake 带来的重计算降低** 多少

## 3. 关键设计澄清（讨论中确立，必须先对齐）

### 3.1 Mooncake：`kv_both` + MooncakeStoreConnector = backing store，但 vllm 0.18 不支持 ⚠️
- 用户确认目标模式：`kv_both` + `MooncakeStoreConnector` = **backing store**（GPU 驱逐→Mooncake→restore 避免重算）。⚠️ `kv_both` **不是** PD 分离/跨副本 transfer（此前误判），是**自存自取的本地二级存储**。`parallel_infer.py:89-100` 的 `kv_role=kv_both` 配置方向是对的。
- **真正根因不是配置，是版本**：**vllm 0.18 不支持 MooncakeStoreConnector**，需 0.21/0.22+ 才原生支持。→ 0.18 下 connector 没接入（找不到/回退 no-op），这才是"router+Mooncake 仅 +4%、KV transfer 未发生"的原因。
- **后果**：D 组（router+Mooncake）在 0.18 **做不了**，需先升级 vllm（verl 兼容性高风险）。
- **待查**：mooncake 进程是否需用户手动拉起？推测原生版本 connector 自管 store，之前 store_service core dump + 装 etcd/redis 是 0.18 硬搞的产物。

### 3.2 开 Mooncake 后，GPU hit rate 可能不升反降 ⚠️
- backing store 模式下，GPU cache 可能更激进地把冷 block 卸到 store（反正能捞回）→ GPU 本地命中率可能下降。
- **但 new_tokens（重算）下降才是真收益**。所以第②件（GPU hit）和第③件（重算降低）可能反向变化——**不能单看 GPU hit 判断 Mooncake 有没有用，要看 new_tokens**。（此洞察对升级后的 D 组成立。）

### 3.3 "重算"指什么：是 new_tokens，不是 num_preempted
- `num_preempted`（preempt 活跃序列，RECOMPUTE 下≈序列重算）对 **prefix LRU eviction 完全失明**，推理场景几乎恒 0，**不作主指标**。
- "复用被打断"= **prefix cache miss → new_tokens 增加**。核心指标 `computed_tokens`（复用）vs `new_tokens`（重算）。
- 前置：开 vLLM `disable_log_stats=False`（memory `kvc-router-disable-log-stats`）拿 StatLogger 统计。

### 3.4 reset_prefix_cache 不是推理场景的元凶（已证伪）
- 读 `vllm_async_server.py:612-650,687,766` 确认：`reset_prefix_cache` 只在 **RL weight-update 边界（wake_up/sleep）+ abort 残留请求**触发，**推理场景轮间/sample间不触发** → prefix 下降元凶是 LRU eviction 本身。

### 3.5 关于"mooncake 拉起还有问题"（旧结论 #3）的新视角
- 之前 mooncake 拉起的各种问题（store_service core dump、装 etcd/redis、bootstrap 端口冲突），很可能是 **0.18 不原生支持、手动编译+手动拉 store_service 硬搞**的产物。
- 升级到 0.21/0.22 原生支持后，connector 可能自管 store，这些拉起问题可能**消失**——需在 P0b 验证。

### 3.6 关于"router 有吞吐无 prefix 收益"（旧结论 #2）
- 结论保留：router（C 组）做的是**调度优化**（同源请求导向能复用 prefix 的副本），prefix 提升有限是预期的——真正的 prefix 收益要靠 Mooncake backing store（D 组）降 new_tokens。
- 所以 router 的 prefix 收益要看 C 组单独量化，不要指望 router 单独解决驱逐。

## 4. 实验设计

### 计划分两腿（因 vllm 版本约束）
- **腿一（ABC 组，0.18 可立即做）**：量化 router vs sticky 的吞吐、router 的 prefix 收益。**不依赖 Mooncake，无版本阻塞。**
- **腿二（D 组，需升级 vllm，独立高风险）**：量化 Mooncake 降重算。**阻塞在 P0b。**

### 固定变量（保证可比）
| 项 | 值 | 理由 |
|---|---|---|
| 模型 | **8B** | 4B KV 太小难触发 eviction；30B 必须 TP>1 → Mooncake 死锁挂不了 |
| 拓扑 | **TP=1 DP=8** | 让 Mooncake 可用（TP>1 必死锁）+ 四组公平对比 |
| 并发 | c=16 | 阶段一~三验证的安全上限 |
| 数据 | SWE-bench 子集 + **n=4 多采样** | 复现轮间+sample间两种复用 + 驱逐 |
| 触发驱逐 | 必要时降 `gpu_mem` 或拉长上下文 | 8B 默认 eviction 可能不够 |

### 四组对比
| 组 | 调度 | Mooncake | 腿 | 量化目标 |
|---|---|---|---|---|
| **A baseline** | vLLM 默认 round-robin | 关 | 一 | 基线 |
| **B sticky** | rollout→副本亲和 | 关 | 一 | vs A：sticky 复用收益 |
| **C KVCAware router** | KVCAwareBalancer | 关 | 一 | vs B：router vs sticky → 目标① |
| **D router+Mooncake** | KVCAwareBalancer | 开（backing store） | **二（需升级）** | vs C：Mooncake 降重算 → 目标③；GPU hit → 目标② |

每组记录：**吞吐(tok/s, req/min)、computed_tokens、new_tokens、GPU prefix hit rate、num_preempted(佐证)、Mooncake restore 次数(D)**。

## 5. 阶段任务

### P0a — ABC 前置（0.18 可做，无阻塞）
- [ ] T0.1 开 `disable_log_stats=False`，确认 StatLogger 打印 computed/new tokens
- [ ] T0.2 实现 **sticky 调度器**（B 组对照）：同 rollout_id 固定副本。落点 `strategies/`

### P0b — D 前置（高风险，独立分支）
- [ ] T0.3 确认 vllm 各版本对 MooncakeStoreConnector 的支持门槛（0.18 不支持，0.21/0.22+ ？）
- [ ] T0.4 评估升级 vllm + verl e546f147 兼容性（verl 对 vllm 敏感，可能需 fork 适配）→ **决定 D 组是否可行 / 排期**
- [ ] T0.5 升级后确认 mooncake 进程是否需手动拉起 / connector 是否自管 store（验证 3.5 推测）
- [ ] T0.6 TP=1 DP=8 验证 backing store 真能 eviction→store→restore（new_tokens 不增）

### 腿一：P1 A → P2 B → P3 C（量化目标①，部分②）
- [ ] 逐组跑，记录全套指标，组间对比

### 腿二：P4 D（量化目标③，部分②）—— 依赖 P0b
- [ ] 升级 vllm 后跑 D，对比 C

### P5 — 报告
- [ ] 四组对比表 + 结论：router 吞吐增益、sticky vs router、Mooncake 降重算幅度、GPU hit 是否提升

## 6. 关键决策记录

- **D1**：固定 TP=1 DP=8 + 8B，四组可比且 Mooncake 可用（牺牲 30B 真实场景换可控性）。
- **D2**：Mooncake = `kv_both` + MooncakeStoreConnector（backing store，**非 PD 分离**）；vllm 0.18 不支持需升级，**D 组独立高风险分支，不阻塞腿一**。
- **D3**：核心指标 `new_tokens`（重算），非 GPU hit rate（D 组可能反降）。
- **D4**：`num_preempted` 仅佐证（推理场景失明）。
- **D5**：腿一（ABC）先行出结果，腿二（D）待 vllm 升级评估——不要让 Mooncake 升级阻塞 router/sticky 的量化。
- **D6（2026-06-22，D 组实际结果，覆盖 D2 的"独立高风险/阻塞"）**：D 组**已突破并跑通（4B）**。路径：vllm 0.21 cu129 源码编译（绕 cu13 wheel）→ mooncake-transfer-engine 装 → `/metadata` endpoint（不需 etcd）→ store.setup 成功 → **4B 完整跑通**（BlockStored 22309 KV 进 store，EngineDeadError=0）。**目标③初步数据**（4B C vs D）：C router-only GPU hit 95-98% vs D router+Mooncake 0-50%波动 → **mooncake 在 KV 不紧张(4B)时 hit 反降，价值在驱逐场景**。8B 因 KV 大 handle pool 耗尽崩（NO_AVAILABLE_HANDLE），待 mooncake 升级。结论：MooncakeStoreConnector 架构可用，价值取决于 KV 紧张度。详见 `experiment_report.md`。

## 7. 环境速查
| 项 | 值 |
|---|---|
| 147 | `root@8.92.9.147`，容器 `hgq-swe`，仓库 `/data1/hgq/uni-agent` |
| 150 | 分析机（8 卡） |
| verl | `e546f147`（vllm_async_server.py 在 **`verl/verl/workers/rollout/vllm_rollout/`**，双层 verl 勿漏） |
| vLLM | **0.18（不支持 MooncakeStoreConnector，D 组需升级 0.21/0.22+）** |
| 模型 | `/data1/models/Qwen/`（实验用 8B） |
| 推理入口 | `scripts/infer_multi.sh` → `examples/agent_interaction/parallel_infer.py` |
| 关键代码 | `parallel_infer.py:82-100`(Mooncake+kv-events) / `strategies/kvc_aware.py`(router 打分) / `balancer.py`(acquire_server) |
| 安全并发 | c=16 / CAP=70000 / nnodes=1 / Ray idle timeout 已修 |

## 8. 注意事项
- 147 操作硬约束：只在 `/data1/hgq` 内、`hgq-swe` 容器内，高危先审批（见 `uni_agent/llm_router/docs/env.md`）
- 代码同步走 git，不走 scp；改动先 WSL commit/push，147 pull
- Mooncake TP>1 死锁是硬约束，D 组锁死 TP=1
- vllm 升级是高风险动作，先在 150 验证 verl 兼容，再动 147
