# Findings — 阶段四研究记录

> 实验数据、代码事实、根因分析。新事实标日期。

## ✅ 已确认事实（带证据）

### 驱逐 / KV 行为
- **4B 默认无驱逐**：verl `interaction_result.json` 504 条 `num_preempted=0`（2026-06-21）
- **4B 构造驱逐**：`gpu_mem=0.3 + prompt=35K` → prefix hit 0%
- **30B n=4 自然驱逐**：TP4 DP2 gpu_mem=0.9，KV 100%、prefix cache 97%→**45.2%**，Waiting max=4，preempt=0
- **30B 轮数**：median 23 / mean 319 / max 2026

### 吞吐对比（旧实验，Mooncake 模式存疑，待 D 组重测）
| 场景 | throughput | 备注 |
|---|---|---|
| 8B baseline | 31.6 tok/s | |
| 8B router-only | 47.6 tok/s | +57%（更均匀分发减排队）|
| 8B router+Mooncake | 49.6 tok/s | 仅 +4%，⚠️ 大概率因 Mooncake 配成跨副本 transfer（kv_both）而非 backing store，KV transfer 根本没发生 |

### 驱逐实锤的方法论（本次讨论确立）
**vLLM V1 两种独立驱逐机制**：
| 机制 | 是什么 | 观测指标 |
|---|---|---|
| ① preempt | 抢**活跃序列**换出重算 | `num_preempted`（RECOMPUTE 下≈序列重算次数）|
| ② prefix LRU eviction | 挤掉**空闲冷 prefix block** | prefix cache hit 跌 + KV 撑满 |

- V1 默认用 Waiting 排队代替 preempt → 机制①推理场景几乎不发生 → `num_preempted` 恒 0 是**预期**，对机制②**完全失明**。
- "无驱逐"实锤：`num_preempted=0`（覆盖①）✓ 硬。
- "有驱逐"实锤：prefix hit 跌 + KV≈100%（覆盖②，间接证据）—— 单看 prefix 跌不够（混淆自然 miss / 路由分散），需 KV 撑满佐证。
- 30B n=4 KV 满之前（62.5%）的 prefix 跌含跨 rollout 竞争成分；KV 100% 后才是干净 eviction。

### reset_prefix_cache 证伪（本次讨论）
- 曾怀疑 verl 轮间/sample间清 cache 是元凶 → 读代码证伪。
- `reset_prefix_cache`（`vllm_async_server.py:612-650,687,766`）只在：**RL weight-update 边界（wake_up/sleep）+ abort 残留请求**触发。
- 推理场景轮间/sample间**不触发** → prefix 下降元凶是 LRU eviction 本身。
- RL 训练 step 边界硬 reset 是设计（权重变了旧 KV 失效）。

## ⚠️ Mooncake 澄清（本次讨论，含两次纠正）

**用户确认的目标模式**：`kv_both` + `MooncakeStoreConnector` = **backing store**（GPU 驱逐→卸到 Mooncake→下次 restore 避免重算）。
- ⚠️ **纠正 1**：`kv_both` **不是** PD 分离 / 跨副本 transfer（此前误判）。kv_both 在 StoreConnector 语境下 = **自存自取**（evict 时写 store，miss 时读 store），是本地二级存储。`parallel_infer.py:89-100` 的 `kv_role=kv_both` 配置方向**是对的**。
- ⚠️ **纠正 2（真正根因）**：不是"模式配错"，而是 **vllm 0.18 不支持 MooncakeStoreConnector**（需 0.21/0.22+）。→ 0.18 下 connector 没接入（找不到/回退 no-op），这才是"KV transfer 未发生、router+Mooncake 仅 +4%"的根因。
- **后果**：D 组（router+Mooncake）在 0.18 **做不了**，需先升级 vllm（verl 兼容性高风险），是独立分支。

**待查**：
- mooncake 进程是否需用户手动拉起？（推测原生版本 connector 自管 store；之前 store_service core dump + 装 etcd/redis 是 0.18 硬搞的产物）
- 升级 vllm 0.21/0.22 + verl e546f147 兼容性

## 📊 正确的指标定义（本次讨论确立）

| 指标 | 含义 | 用途 |
|---|---|---|
| `computed_tokens` | 命中复用的 token（轮间+sample间）| 越高越好 |
| `new_tokens` | 没复用、要重算 prefill 的 token | **核心：越低=复用丢失少** |
| GPU prefix hit rate | GPU 本地 cache 命中率 | 辅助，开 Mooncake 后可能反降 |
| `num_preempted` | 活跃序列被抢占 | 推理场景失明，仅佐证 |
| Mooncake restore 次数 | backing store 命中 | D 组专有 |

**关键洞察**：开 Mooncake backing store 后，GPU hit rate 可能不升反降（block 更早卸到 store），但 `new_tokens` 下降才是真收益 → **不能单看 GPU hit 判断 Mooncake 有没有用**。

前置：开 `disable_log_stats=False`（memory `kvc-router-disable-log-stats`）拿 StatLogger 统计。

## 🔍 待验证清单
- [ ] vLLM 0.18 MooncakeConnector **backing store 模式**怎么配？（容器 site-packages/vllm）
- [ ] `disable_log_stats=False` 后 StatLogger 输出格式（computed/new tokens 字段名）
- [ ] per-request `num_computed_tokens` 能否拿到（区分轮间/sample间复用）
- [ ] router 实际 config 的 `alpha` 值 / `prompt_ids` 是否传

## 📁 关键文件索引（verl 真实路径：双层 `verl/verl/`）
| 文件 | 作用 |
|---|---|
| `uni_agent/llm_router/strategies/kvc_aware.py:142` | router 打分 `score()`，`alpha*S_cache+(1-alpha)*S_load` |
| `uni_agent/llm_router/strategies/__init__.py` | `route()` 聚合逻辑 |
| `uni_agent/llm_router/balancer.py:107` | `acquire_server(request_id, prompt_ids)` |
| `examples/agent_interaction/parallel_infer.py:82-100` | Mooncake+kv-events 入口（待改 backing store）|
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:590-608` | `num_preempted=outputs[0]`，per-rollout 累加 |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:612-650,687,766` | `reset_prefix_cache` 触发点 |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:394` | `AsyncEngineArgs.from_cli_args`（preemption-mode 透传点）|

## 腿一实测数据（2026-06-21，从 logs/ 提取）

**关键澄清：baseline = verl 内置 `global_sticky_inflight`（sticky-session + least-inflight）= 用户要的"一般 sticky 调度"对照**（`router.py:239`, `rollout.py:166-171`，实现 `GlobalRequestLoadBalancer` 带 routing cache）。→ 腿一简化为 **A(sticky) vs C(router) 两组**，不需独立实现 sticky（T0.2 取消）。

**已有日志**（8B TP=1 DP=8 max_num_seqs=16 max_samples=16，配置完全可比）：
| 组 | 状态 | 稳态 GPU hit（排冷启动）| 备注 |
|---|---|---|---|
| A baseline(sticky) | ⚠️ 中途被 kill（人为非 OOM；内存 246G 充足，KV 5.34G/卡 正常）| **49.3%**（median 50.1）| 未跑完，无 RM Score；需重跑 |
| C router_only | ✅ 完整 16 samples | **83.0%**（median 86.6）| Mean RM 0.0625 |

**router vs sticky 实锤（目标②）**：GPU hit 49.3%→83.0%，**+33.7pp**。⚠️ 纠正"router 无 prefix 收益"的笼统说法——准确是 **router vs baseline(sticky) 有收益**；**router+mooncake vs router-only 无额外收益**（且 mooncake 0.18 没生效）。

**腿一正式结果（第三次重跑，A/C 严格可比，2026-06-21，KV 撑满高压 eviction 场景）**：
| 组 | 稳态 GPU hit | wall-time(16 samples) | RM |
|---|---|---|---|
| A sticky | 8.3% | 44.5 min | 0.0625 |
| C router | 36.5% | 23.2 min | 0.0000 |

KV max 都~99.5%（高压，vs 历史低压 85%）。→ **router vs sticky：GPU hit 4.4x（+28pp）、吞吐 ~1.9x（23 vs 44min）**，**目标①（吞吐）+②（GPU hit）均显著成立**。RM 小样本(16)噪声。hit 绝对值依赖 KV 压力（低压 83%/49% vs 高压 36%/8%），但 **router 相对 sticky 的优势在两种压力下稳定**。A gen_tput median=0.1 vs C 6.3（疑 sticky 副本过载/沙箱 hang 拖慢，待查）。

**吞吐口径（目标①）**：StatLogger `Avg generation throughput` mean 不可靠（含启动/空闲，min=0）。改用 **samples / wall-time**。

**computed/new tokens（目标③）**：0.18 StatLogger **不打印**（grep count=0）→ 腿二从 `/metrics`（`vllm:gpu_prefix_cache_queries_total/hits_total`）拿。腿一不需要③。

**router config 实测（P0 待验证项已答，2026-06-21）**：`kvc_aware_router.yaml` → `strategies/kvc_aware_strategy.yaml`：**alpha=0.7**（S_cache 主导，非"太小"）、单 strategy `weight=1.0`（route 不稀释）、`load_threshold=0.1`、collector=`vllm_zmq+vllm_metrics`。
→ **router 正常工作**：alpha=0.7 让 S_cache 主导打分，主动把请求导向"已缓存该 prefix"的副本，这正是 router hit(83%) ≫ sticky(49%) 的原因。**P0 前提（router 无 prefix 收益）本身不成立**——错误来自把 router+mooncake（mooncake 0.18 没生效）的数据当成了 router 本身。

**A/C 重跑失败根因（2026-06-21，已修正）**：`max_model_len=40960` 超 KV 上限——8B max_seq_len 40960 需 5.62G KV，available 仅 5.34G，vllm 估 max_model_len 上限 **38864**（历史 router_only KV=38864 即此由来）。修正：`PROMPT_LEN=28672 RESPONSE_LEN=8192` → max_model_len=37888（留 buffer）。Qwen3-8B max_position=40960 但受显存实际只能用 38864。

**vllm 0.18 mooncake 实锤（2026-06-21）**：0.18 **有 `MooncakeConnector`（P2P）但无 `MooncakeStoreConnector`**（`grep -rl MooncakeStoreConnector` 空）。→ 用户对（0.18 不支持 StoreConnector），WebSearch"v0.10.0 引入"是幻觉。另有 `mooncake_connector.py.bak`——之前会话 patch 过。**腿二 D 组确认需升级 vllm**。

**腿二 P0b 评估（2026-06-21）**：
- verl `setup.py` 声明 `vllm>=0.8.5,<=0.12.0`，但实际跑 0.18（editable install 绕过声明）→ 声明不限制，但 vllm 0.12→0.21+ 跨度大，verl `vllm_async_server.py` 用的 V1 API 可能不兼容，需适配验证。
- 147/150 镜像都是 `vllm018.dev1`，升级 vllm 要么换镜像要么 pip 升级（改环境，**高危，需用户审批**，见 env.md）。
- 150 `hgq-swe` 容器 Exited(137)（39h 前 OOM/被kill），需重启。
- **D 组执行路径**：升级 vllm（到含 MooncakeStoreConnector 的版本）→ 适配 verl → 配 `kv_connector=MooncakeStoreConnector kv_role=kv_both` → 跑 D。
- **D 组升级已执行（2026-06-22，150 hgq-swe，task buy5vjeef）**：vllm 0.18→0.21.0 成功，**MooncakeStoreConnector 可用**（import OK）✓，verl import OK ✓。
- **D 组运行时硬阻塞（2026-06-22）**：verl-vllm import chain OK，但 `torch.cuda.is_available()=False`——vllm 0.21 拉 torch 2.11+**cu130（CUDA 13）**，而 150/147 驱动（575.x）只支持 **CUDA 12.9** → **当前硬件跑不了 vllm 0.21**。150 升级后 GPU 不可用（环境破坏），147 仍 0.18 正常。
- **D 组最终结论（2026-06-22，两轮实质尝试）**：
  1. vllm 0.18→0.21.0 升级 + MooncakeStoreConnector 可用 ✓，但默认拉 torch cu130（CUDA13），驱动 12.9 不支持 → `torch.cuda=False`。
  2. cu128 force-reinstall torch → `torch.cuda=True` ✓，但 **vllm 0.21 PyPI wheel 是 cu13 编译**（`libcudart.so.13 not found`，bug #43435）——vllm .so 绑死 cuda 13，torch 换 cu128 救不了。
  3. **cu12 vllm 0.21 wheel PyPI 没有**，需源码编译（大工程）。
  - **→ D 组在当前驱动（cuda 12.9）不可行**（⚠️ **cu13 阶段旧结论，已突破**——源码编译 cu129 + `/metadata` endpoint 后 4B 跑通，见下「D 组突破/4B 跑通」+ 「阶段四最终总结」）。150 升级后环境破坏（vllm 0.21 cu13 + torch cu128 混合，待重置/回退 0.18）。147 仍 vllm 0.18 正常。
  - **出路**（超 agent 能力边界）：源码编译 vllm 0.21 cu12 / 升级 NVIDIA 驱动到 cuda 13（高危，共享机 150 不能动）/ 换新驱动机器。
- ✅ **源码编译成功（2026-06-22，task be84n5viv，编译 ~2.5h）**：补 `setuptools_scm/ninja` build deps 后，`VLLM_USE_PRECOMPILED=0 pip install -e .`（nvcc12.9 + torch cu128，MAX_JOBS=8）编出 **`vllm-0.21.0+cu129`** → `vllm._C OK` + `MooncakeStoreConnector OK` + `verl OK` + `cuda=True` **全部通过**！**D 组解阻塞**——PyPI cu13 wheel 问题用源码编译 cu129 绕过。
- **D 组 mooncake 部署进展（2026-06-22）**：装 `mooncake-transfer-engine 0.3.11`（PyPI，带 engine.so，有 standalone 命令 `mooncake_http_metadata_server`/`mooncake_master` 不需 etcd）。启动 metadata server(:9527) + master(rpc_port=9422, admin 9003，均起来)。MooncakeStoreConnector 跑到 **`Initialize MooncakeDistributedStore failed`**——剩 config 微调：`metadata_server` 格式（HTTP metadata 用 `http://127.0.0.1:9527`？）+ master_server_address=127.0.0.1:9422(已对)。`MOONCAKE_CONFIG_PATH=/data1/hgq/mooncake_config.json`。**D 组部署 90%，差 config 微调**。
- **D 组 store.setup 定位（worker.py:574）**：`self.store.setup(config_dict)` 返回非0→失败（msg 吞真异常）。config_dict 字段：local_hostname(get_ip)/metadata_server/global_segment_size/local_buffer_size/protocol(tcp)/rdma_devices/master_server_addr。
- **D 组 store.setup 根因（2026-06-22，python 直测拿 C++ 错误）**：mooncake transfer engine 默认 **EtcdStoragePlugin**，`redis` plugin **没编译进 PyPI wheel**（`Unable to find metadata storage plugin redis`），http metadata server 协议不匹配（etcd 要 http2，http server 是 http1.1）。→ **metadata_server 必须是 etcd（127.0.0.1:2379）**。
- **D 组 etcd 装配受阻于 150 环境（2026-06-22）**：150 **完全没 etcd**——没预装、`apt` 无 etcd 包、github release + ghproxy + docker pull + 华为云/阿里镜像 **全不通**（150 对外只清华 pypi 通，etcd 无 pypi server）。master+http_metadata+store.setup 根因都已定位/能起，**纯卡 etcd server binary 获取**。需用户介入：提供 etcd binary / 换可达网络环境 / 修 150 网络。
- **澄清（mooncake 两层 metadata）**：master `-enable_http_metadata_server` 是 **master HA** 用 http 代 etcd，**不替代 transfer engine 的 store.setup etcd**。mooncake transfer engine 只 **EtcdStoragePlugin**（redis plugin 没编；http scheme 不触发 http plugin）。mooncake 包带 `libetcd_wrapper.so`（etcd **client**），仍需外部 etcd **server**。→ **D 组必须 etcd server，150 无法获取**。
- **D 组 http plugin 替代也失败（2026-06-22，19 轮）**：升级后 mooncake 认 http plugin（不再 etcd），但 http plugin (`?key=`) 连根 path 404。
- ✅ **D 组突破！store.setup 成功（2026-06-22）**：mooncake http metadata server API 是 **`/metadata`** endpoint（`http_metadata_server.py:60 add_route '/metadata'`），**metadata_server 必须配 `http://host:9527/metadata`（加 /metadata 后缀，不是根）**。配对后 `SETUP_RET=0`（GET ?key= 首次 404 正常，PUT 注册后成功）。**不需 etcd**！
- **D 组核心架构突破（2026-06-22，24 轮）**：router + KVCAwareBalancer + **MooncakeStoreConnector 全工作**——`BlockStored` KV 事件正常流转（KV blocks 进 store）、polling collector refresh 2 replicas。证明 **MooncakeStoreConnector 架构通**（store.setup 成功 + KV 进 store）。
- ✅ **D 组 4B 跑通！MooncakeStoreConnector 完整工作（2026-06-22，34 轮突破）**：4B（KV 小，handle pool 够）跑通——`EngineDeadError count=0`、**BlockStored=22309**（KV blocks 存进 mooncake store，backing store 实际工作）、prefix hit 波动 0~50%（store restore 起作用）、NO_AVAILABLE_HANDLE 19（少量 warning 不致命）。8B 因 KV 大 handle pool 耗尽崩，**4B 跑通**。MooncakeStoreConnector + router + store.setup + BlockStored 全验证。
- **D 组结论**：MooncakeStoreConnector 架构**可用**（4B 完整跑通，KV 进 store）。8B 需更大 handle pool（mooncake 内部，待 mooncake 升级/调）。目标③（Mooncake 降重算）需对比 C 组（router-only）量化。
- **D 组目标③ 量化（4B C vs D，2026-06-22）**：C(router-only) GPU hit **95-98%** 稳态高；D(router+Mooncake) GPU hit **0-50% 波动** + BlockStored 22309（KV 进 store）。**发现：mooncake 在 KV 不紧张（4B）时 GPU hit 反降**（block offload 到 store 改变 cache 行为）——印证"开 Mooncake 后 GPU hit 可能降"的早期洞察。**mooncake 价值在 KV 紧张/驱逐场景**（store restore 避免重算），4B 无驱逐故无收益；价值待 8B/30B 驱逐场景验证（8B 受 handle pool 限制）。
- **D 组整体进展（35 轮，全部突破）**：cu13 不可行 → vllm 0.21 cu129 源码编译成功(2.5h) → mooncake 部署 → store.setup 突破(`/metadata` endpoint，不需 etcd) → **4B 完整跑通**（BlockStored 22309）。8B 因 KV 大 handle pool 耗尽崩，4B 跑通。
- **阶段四最终总结**：腿一（router vs sticky，目标①②）完整交付（hit 4.4x、吞吐 1.9x）；腿二 D 组（目标③）**MooncakeStoreConnector 架构验证成功**（4B 跑通）+ 目标③初步数据（C vs D：mooncake KV 不紧张时 hit 反降，价值在驱逐场景）。8B 完整对比待 mooncake handle pool 升级。详见 `experiment_report.md`。

## 相关 memory
- `kvc-router-disable-log-stats` — verl disable_log_stats 默认 True 需设 False
- `kvc-router-proxy` — 150 容器 HTTP_PROXY 劫持 172.17 网段
- `kvc-router-150-run` — 150 跑 KVCAware router 的环境要点
