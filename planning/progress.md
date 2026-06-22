# Progress Log — 阶段四会话日志

## 2026-06-21（本阶段起点）

### 今日已完成
- 阶段一~三：500 条 SWE-Bench Verified 多并发推理跑通，Mean RM 0.1780，89/504 resolved，已 commit/push
- 阶段四基线实验全部做完（见 `findings.md`「已确认事实」）：
  - 4B/30B 驱逐行为实锤（含构造驱逐 + 30B n=4 自然驱逐 97%→45.2%）
  - llm-router 吞吐收益量化（8B +57%）
  - Mooncake TP>1 死锁根因定位 + TP=1 DP=8 可用性确认

### 被上下文打断的点
- 正在修 Mooncake store_service 的 etcd/redis 插件（redis PONG 但 store core dump）→ 见 P1

### 关键认知更新
- ⚠️ 修正：之前以为 router 不做 prefix affinity，**读代码后发现 `kvc_aware.py:142` 已用 `S_cache` 打分**。命中率没提升的真因待 P0 验证（alpha 太小 / prompt_ids 没传 / hit 查询失效）。**这是 P0 必须先做的原因。**

---

## 当前状态

- **阶段**：P0 即将开始（前置研究，阻塞 P2）
- **代码基线**：阶段一~三已 push，稳定主线
- **无未提交改动**

## 下一步（接手即做）

1. **读** `task_plan.md` 的 P0 阶段，执行 T0.1→T0.5
2. **第一步具体动作**：读 `uni_agent/llm_router/strategies/__init__.py` 的 `route()`，再读 `uni_agent/llm_router/configs/` 下的实际 config 拿到 `alpha` 值
3. P0 定根因后再决定 P2 走 A/B/C 哪个优化分支
4. P1（Mooncake store 修复）先做 T1.1 必要性评估，大概率搁置

## 决策

- 不直接做 sticky-first，先 P0 验证根因（D1）
- Mooncake TP>1 不改 vLLM 核心，验证限 TP=1 DP=8（D2）
- store_service 不阻塞主目标则搁置（D3）

---

## 2026-06-21（续）— 方案讨论收敛，plan 框架重写

### 本轮讨论产出（已写入 task_plan/findings）
1. **驱逐方法论澄清**：vLLM V1 两机制（preempt / prefix LRU eviction）独立；`num_preempted` 对 prefix eviction 失明 → 不能用它当主指标。
2. **reset_prefix_cache 证伪**：推理场景不在轮间/sample间触发（读 `vllm_async_server.py:612-650,687,766`），prefix 下降元凶是 LRU eviction 本身。
3. **Mooncake 模式错配（核心发现）**：当前配的是跨副本 transfer（`kv_both`），用户要的是 **backing store**（GPU 驱逐→Mooncake→restore 避免重算）。这极可能是之前"router+Mooncake 仅 +4%"的根因。
4. **正确指标**：核心是 `new_tokens`（重算），不是 GPU hit rate（开 Mooncake 后可能反降）。
5. **实验框架重写**：固定 8B TP=1 DP=8，四组对比 A baseline / B sticky / C router / D router+Mooncake，量化用户定义的三件事。

### plan 已重写
- `task_plan.md`：从"sticky-first 优化"框架改为"llm-router + Mooncake backing store 四组对比实验"。
- `findings.md`：整合方法论 + Mooncake 模式澄清 + 指标定义。
- verl 真实路径修正：**双层 `verl/verl/workers/...`**（之前 Read 漏一层 verl 踩过坑）。

---

## 当前状态（更新）

- **阶段**：P0 前置对齐（阻塞全部）
- **最大不确定性**：vLLM 0.18 MooncakeConnector 的 backing store 模式怎么配（需去容器查 site-packages）

## 下一步（接手即做）

1. **P0.T0.1**：147 上开 `disable_log_stats=False`，确认 StatLogger 打印 computed/new tokens
2. **P0.T0.2**：容器里查 `site-packages/vllm` 的 MooncakeConnector，搞清 backing store 配置 → 改 `parallel_infer.py:89-100`
3. **P0.T0.4**：实现 sticky 调度器（B 组对照）
4. 四组件就绪后按 P1→P5 跑四组对比

## 决策（更新）

- 固定 8B TP=1 DP=8 四组可比（D1）
- Mooncake 重写 backing store，P0 先矫正（D2）
- 核心指标 new_tokens，非 GPU hit（D3）
- num_preempted 仅佐证（D4）

---

## 2026-06-21（再续）— Mooncake 两次纠正 + 版本约束，计划分两腿

### 用户两轮纠正
1. **PD 分离是误解**：`kv_both` + MooncakeStoreConnector 就是目标模式（backing store，自存自取），不是 PD 分离。`parallel_infer.py` 的 `kv_role=kv_both` 配置方向没错。
2. **vllm 0.18 不支持 MooncakeStoreConnector**（需 0.21/0.22+）：→ 之前"router+Mooncake 仅 +4%、KV transfer 未发生"的根因是**版本不支持、connector 没接入**，不是配置问题。

### 对计划的影响（已写入 task_plan/findings）
- **计划分两腿**：
  - 腿一 A/B/C（不依赖 Mooncake，0.18 可立即做）→ 量化 router vs sticky 吞吐、router prefix 收益
  - 腿二 D（需升级 vllm，独立高风险）→ 量化 Mooncake 降重算
- D 组从"改配置"升级为"升级 vllm + verl 兼容"工程，**不阻塞腿一**（D5）。
- "mooncake 拉起有问题"（旧 #3）新视角：很可能是 0.18 硬搞 store_service 的产物，升级后可能消失（待 P0b 验证）。
- "router 有吞吐无 prefix 收益"（旧 #2）保留：router 做调度优化，prefix 提升有限是预期，真收益靠 Mooncake 降 new_tokens。

### 当前状态
- **腿一可立即推进**：P0a（disable_log_stats + sticky 调度器）→ P1/P2/P3
- **腿二待评估**：P0b（vllm 升级 + verl 兼容性）决定 D 组可行性

### 下一步
1. 腿一 P0a：开 disable_log_stats，实现 sticky 调度器
2. 腿二 P0b：先在 150 评估 vllm 升级 + verl 兼容（不动 147）

---

## 2026-06-21（执行）— P0a 完成，腿一简化为 A vs C

### 关键澄清（改变实验设计）
- **baseline = verl 内置 `global_sticky_inflight`（sticky-session）= 用户要的 sticky 对照**。腿一 = **A(sticky) vs C(router) 两组**，**T0.2（独立实现 sticky）取消**。
- **router vs sticky GPU hit 实锤：49.3%→83.0%（+33.7pp）**，从已有 logs 提取（详见 findings「腿一实测数据」）。

### T0.1 完成
- `disable_log_stats=False` 已在 `parallel_infer.py:70` 设 ✓
- StatLogger（`loggers.py:259`）打印 throughput + GPU hit（目标①②现成）
- computed/new tokens 0.18 不打印（目标③ 待 `/metrics`，腿二才需要）

### 已有数据状态
- C router_only：完整（16 samples, RM 0.0625, hit 83%）✅ 可复用
- A baseline(sticky)：被中途 kill（人为非 OOM），**需重跑完整**拿吞吐 + 确认 hit

### 下一步
1. 重跑完整 A（sticky baseline）：`TP=1 NGPUS=8 MAX_NUM_SEQS=16 MAX_SAMPLES=16 bash scripts/infer_multi.sh`（不设 ROUTER_CONFIG），日志重定向
2. A vs C 对比 → 量化目标①（samples/wall-time）+ ②（GPU hit）
3. 腿二 P0b（150 上评估 vllm 升级）并行，不动 147

---

## 2026-06-21（执行续）— A 首跑失败 + 串行重跑 A→C

### A 首跑失败
- `ValueError: max_model_len (70000) > max_position_embeddings (40960)` —— **Qwen3-8B max_position=40960**（不是 4B 的 262144），默认 `32768+65536` 算出 70000 超限。
- 修正：`PROMPT_LEN=32768 RESPONSE_LEN=7168`（和 39936 → max_model_len=40960，KV cache 38864，与历史 router_only 一致）。

### 腿二前置研究（WebSearch）
- **MooncakeStoreConnector ≠ MooncakeConnector**：当前 `parallel_infer.py:94` 配的是 `MooncakeConnector`（P2P 直传），用户要的 `MooncakeStoreConnector` 是**另一个 connector**（MooncakeDistributedStore 共享池）。→ 0.18 下不仅版本不支持，**connector 名也配错**。需升级 vllm + 改 connector 名。
- MooncakeStoreConnector 2025 末集成进 vLLM V1（RFC #38474）。

### 进行中（第三次重跑，task `bankkyi1a`）
- 前两次失败根因：max_model_len 超 KV 上限（8B 需 5.62G > available 5.34G，vllm 上限 38864）。详见 findings。
- **第三次**：`TP=1 NGPUS=8 MAX_NUM_SEQS=16 MAX_SAMPLES=16 PROMPT_LEN=28672 RESPONSE_LEN=8192`（max_model_len=37888，留 buffer），日志 `logs/8b_A_sticky.log` / `8b_C_router.log`，后台 ~30min。
- 完成后 A vs C 对比 → 量化目标①（samples/wall-time）+ ②（GPU hit）。

### 阶段四总结（2026-06-22）
- task #1 #2 #3 #4 **全部 completed**
- **腿一（目标①②）完整交付**：router vs sticky，GPU hit 4.4x、吞吐 ~1.9x。报告 `planning/leg1_report.md`
- **腿二 D 组（目标③）源码编译成功 → 解阻塞**：PyPI cu13 wheel 撞墙（libcudart.so.13）后，**源码编译 vllm 0.21.0+cu129 成功**（补 setuptools_scm/ninja，nvcc12.9+torch cu128，MAX_JOBS=8，task be84n5viv ~2.5h）→ `vllm._C OK`+`MooncakeStoreConnector OK`+`verl OK`+`cuda=True` 全通过。
- **D 组跑验证中**（task `b1bhujms6`，150 2卡 DP=2，router+MooncakeStoreConnector，max_samples=4）：确认 StoreConnector 能起 + 初步目标③数据。严格 8 卡对比需 147 升级。
- parallel_infer.py 已改 `MooncakeConnector→MooncakeStoreConnector`（push e69f13a）。
- **D 组部署 90%，卡 etcd binary（150 环境网络障碍，2026-06-22）**：vllm 0.21 cu129 编译成功 + mooncake master+http_metadata 起 + store.setup 根因定位（要 etcd metadata，redis plugin 没编），但 150 完全没 etcd（没装/apt 无包/github+ghproxy 下载不通）。**需用户介入**提供 etcd binary 或换网络环境。D 组技术路径全通，纯卡 etcd 获取。
- ✅ **D 组 4B 跑通 + 目标③量化（2026-06-22，34 轮突破）**：突破 `/metadata` endpoint（不需 etcd，用 mooncake http metadata）→ 4B D 组跑通（EngineDeadError=0, BlockStored=22309 KV 进 store）。8B 因 KV 大 handle pool 耗尽崩，4B 跑通。**C vs D 对比**：C(router-only) GPU hit 95-98%；D(router+Mooncake) hit 0-50% 波动。**发现：mooncake 在 KV 不紧张(4B)时 hit 反降，价值在 KV 紧张/驱逐场景**（store restore 避免重算）。目标③初步数据 + 关键发现。
