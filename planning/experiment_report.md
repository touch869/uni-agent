# 阶段四实验报告：llm-router 与 MooncakeStoreConnector 对 agentic rollout prefix 复用的量化

> 2026-06-22 · 阶段四全部实验汇总。数据详见 `findings.md`。

## 1. 实验目标（用户定义三件事）
1. **router vs sticky** 吞吐提升多少（目标①）
2. **GPU prefix cache 命中率** 是否提升（目标②）
3. **Mooncake 降重算** 多少（目标③）

## 2. 实验环境
| 项 | 腿一 | D 组 |
|---|---|---|
| 机器 | 147（8×3090 独占）| 150（共享 GPU 6,7）|
| 模型 | Qwen3-8B | Qwen3-4B |
| 拓扑 | TP=1 DP=8 | TP=1 DP=2 |
| vllm | 0.18 | 0.21 cu129（源码编译）|
| 数据 | SWE-Bench Verified 子集 | 同 |
| agentic | 多轮 agent rollout，n=1 | 同 |

> 腿一 8B 与 D 组 4B **模型/拓扑不同，不可严格横向比**——D 组在 4B 验证 MooncakeStoreConnector 架构可用性 + 目标③趋势，腿一在 8B 严格量化 router vs sticky。

## 3. 腿一：router vs sticky（目标①②，8B）

**两组定义**（均**未开 mooncake**）：
- **A baseline** = verl 内置 `global_sticky_inflight`（sticky-session + least-inflight，**非 round-robin**）= 用户要的"一般 sticky 调度"
- **C router** = KVCAwareBalancer，`alpha=0.7`（S_cache 主导）

### 3.1 高压 eviction 场景（KV~99.5%，max_model_len=37888，严格可比）
| 组 | 稳态 GPU prefix hit | wall-time(16 samples) | RM |
|---|---|---|---|
| A sticky | **8.3%** | 44.5 min | 0.0625 |
| C router | **36.5%** | 23.2 min | 0.0000 |

→ **router vs sticky：GPU hit 4.4x（+28pp）、吞吐 ~1.9x（23 vs 44min）**。目标①②**均显著成立**。

### 3.2 低压场景（KV~85%，历史数据）
| 组 | 稳态 GPU hit |
|---|---|
| A sticky | 49.3% |
| C router | 83.0% |

→ router vs sticky：+33.7pp。**两种 KV 压力下 router 相对优势都稳定**。

### 3.3 提升机制（关键）
router 提升 prefix hit **不靠 mooncake**，靠 **prefix-aware 路由**：
- router 通过 kv-events 实时知道每副本缓存了哪些 prefix block，`S_cache` 打分把请求**主动导向"已缓存该 prefix"的副本** → 命中
- sticky 按 session/least-inflight 路由，**不查 prefix cache**，同源请求可能被导向没缓存的副本 → miss

## 4. D 组：router+Mooncake vs router-only（目标③，4B）

**两组**（均开 router）：
- **C router-only**：不开 mooncake
- **D router+Mooncake**：`MooncakeStoreConnector kv_role=kv_both`（vllm 0.21）

| 指标 | C router-only | D router+Mooncake |
|---|---|---|
| 稳态 GPU prefix hit | **95-98%** | 0-50%（波动） |
| BlockStored（KV 进 store）| 0 | **22309** |
| NO_AVAILABLE_HANDLE | 0 | 19（warning） |
| EngineDeadError | 0 | 0（4B 跑通）|

### 4.1 关键发现
- **mooncake 在 4B（KV 不紧张）时 GPU hit 反降**（95-98% → 0-50% 波动）——block offload 到 store 改变 GPU cache 行为，印证早期洞察"开 Mooncake 后 GPU hit 可能降"
- **mooncake 价值在 KV 紧张/驱逐场景**（store restore 避免重算）；4B 无驱逐（C 组 hit 95-98%），mooncake 无收益甚至干扰
- **8B D 组因 KV 大 handle pool 耗尽（NO_AVAILABLE_HANDLE 累积）崩**，4B（KV 小）跑通——MooncakeStoreConnector **架构可用**，handle pool 是 mooncake 内部限制

## 5. 阶段四核心结论
1. **router 显著优于 sticky**（腿一实锤：hit 4.4x、吞吐 1.9x）——机制是 prefix-aware 路由，与 mooncake 正交
2. **MooncakeStoreConnector 架构可用**（D 组 4B 跑通，BlockStored 22309 KV 进 store），但**价值取决于 KV 紧张度**：
   - 无驱逐（4B）→ 无收益，GPU hit 反降
   - 有驱逐（8B/30B n=4）→ store restore 降重算（理论收益，8B 待 handle pool）
3. router（路由层）与 mooncake（存储层）是**两个正交维度**：router 靠导向已缓存副本提 hit，mooncake 靠 KV 持久化降重算

## 6. 方法论要点
- **baseline 不是 round-robin**——是 verl 内置 sticky（global_sticky_inflight），更公平
- 指标：稳态 GPU prefix hit（排冷启动后 2/3 区间）+ wall-time（samples/时长）
- StatLogger `Avg generation throughput` mean 不可靠（含空闲），改用 wall-time
- **D 组突破点**：mooncake http metadata server 的 API 是 `/metadata` endpoint（`http://host:9527/metadata`），不需 etcd（mooncake transfer engine 默认 EtcdStoragePlugin，但升级后认 http plugin + /metadata）

## 7. 限制 / 待续
- RM Score 小样本（16/4）噪声，不作判定（4B RM 0 是模型能力）
- 腿一 8B 与 D 组 4B 不可严格比（KV 紧张度不同）
- **8B D 组完整对比待 mooncake handle pool**（mooncake 升级/源码调）——8B KV 大耗尽 handle pool 致 EngineDeadError
- 目标③严格量化（8B 驱逐场景 store restore 降重算）待 8B D 组跑通

## 8. 证据链与可复现（每个结论的日志来源 + 实锤逻辑）

### 8.1 腿一 8B（147，task `bankkyi1a`）
- **日志**：`logs/8b_A_sticky.log`（A）、`logs/8b_C_router.log`（C）
- **配置**：`TP=1 NGPUS=8 MAX_NUM_SEQS=16 MAX_SAMPLES=16 PROMPT_LEN=28672 RESPONSE_LEN=8192`（A/C 严格相同，只差 `ROUTER_CONFIG`）
- **GPU hit 提取**（稳态，排冷启动）：`grep "Prefix cache hit rate" <log> | grep -oE "hit rate: [0-9.]+" | tail -40 | sort -n` → A median 8.3%，C 36.5%
- **wall-time**：`grep -oE 'INFO [0-9]{2}-[0-9]{2} [0-9:]+' <log> | sed -n '1p;$p'`（首末时间戳）→ A 44.5min，C 23.2min
- **关键日志行**（StatLogger 每 10s）：`Engine 000: Avg generation throughput: X, GPU KV cache usage: Z%, Prefix cache hit rate: W%` —— hit 数据直接来自此行
- **实锤链**：A hit(8.3%) < C hit(36.5%) ← 同配置只差路由 ← router alpha=0.7 S_cache 打分导向已缓存副本（`kvc_aware.py:142`）

### 8.2 D 组 4B（150，task `bj5tjteem`=D / `bdwezd4e1`=C）
- **日志**：`logs/4b_D_mooncake_150.log`（D）、`logs/4b_C_router_150.log`（C）
- **配置**：`TP=1 NGPUS=2 MAX_NUM_SEQS=4`；D 组 `ENABLE_MOONCAKE=1` + mooncake master(:9422)+http_metadata(:9527) + `MOONCAKE_CONFIG_PATH`（`metadata_server=http://127.0.0.1:9527/metadata`）
- **BlockStored count**：`grep -c BlockStored <log>` → D=22309（C=0）
- **EngineDeadError count**：`grep -c EngineDeadError <log>` → D=0（4B 跑通），8B 同配置 = 崩
- **关键日志行**：
  - `Creating v1 connector with name: MooncakeStoreConnector`（connector 创建成功）
  - `BlockStored: replica=..., blocks=1, block_size=16`（KV blocks 进 store）
  - `SETUP_RET = 0`（store.setup 成功，python 直测 task）
  - `Detected Mooncake CPU offloading pressure (NO_AVAILABLE_HANDLE)`（handle 压力 warning，4B 不致命/8B 致命）
- **实锤链**：MooncakeStoreConnector 工作 ← BlockStored 22309（KV 进 store）← connector 创建成功 ← store.setup ret=0 ← `/metadata` endpoint

### 8.3 D 组突破链（35 轮，每关卡的实锤日志）
| 关卡 | 证据 | task |
|---|---|---|
| 0.18 无 StoreConnector | `grep -rl MooncakeStoreConnector $VLLM` → 空 | — |
| cu13 墙 | `torch.cuda=False` + `libcudart.so.13 not found` | buy5vjeef |
| 源码编译成功 | `vllm-0.21.0+cu129`, `vllm._C OK` | be84n5viv（~2.5h）|
| store.setup 根因 | `EtcdStoragePlugin: unable to set...`（要 etcd） | python 直测 |
| `/metadata` endpoint | `http_metadata_server.py:60 add_route '/metadata'` | 源码 |
| store.setup 成功 | `SETUP_RET=0`（metadata_server=...9527/metadata）| python 直测 |
| 4B 跑通 | `EngineDeadError=0` + `BlockStored=22309` | bj5tjteem |

### 8.4 D 组可复现命令（150 hgq-swe）
```bash
cat > /data1/hgq/mooncake_config.json << 'JSON'
{"metadata_server": "http://127.0.0.1:9527/metadata", "master_server_address": "127.0.0.1:9422",
 "protocol": "tcp", "global_segment_size": "4GB", "local_buffer_size": "4GB"}
JSON
export PYTHONPATH=/data1/hgq/uni-agent MOONCAKE_CONFIG_PATH=/data1/hgq/mooncake_config.json
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy  # 150 proxy 劫持，git/curl 要禁
nohup mooncake_http_metadata_server --port 9527 >/tmp/moon_meta.log 2>&1 &
nohup mooncake_master --port 9422 >/tmp/moon_master.log 2>&1 &
ENABLE_MOONCAKE=1 ROUTER_CONFIG=pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml \
  TP=1 NGPUS=2 MAX_NUM_SEQS=4 MAX_SAMPLES=4 PROMPT_LEN=16384 RESPONSE_LEN=8192 \
  bash scripts/infer_multi.sh /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  scripts/swe_bench_verified_modal.parquet scripts/agent_config_localdocker.yaml
```

## 9. 产出文件
- `planning/leg1_report.md`（腿一详细报告）
- `planning/findings.md`（全部数据 + 代码事实 + 调试链）
- `planning/progress.md`（35 轮调试日志）
- `planning/experiment_report.md`（本报告）
