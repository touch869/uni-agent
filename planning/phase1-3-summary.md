# 阶段一~三 工作总结（已 commit & push）

> 2026-06-21 · uni-agent × SWE-Bench Verified 多并发推理调通
> 阶段四的 plan 见同目录 `task_plan.md`，本文为前三阶段归档。

## 目标

在 147 机器（8×RTX 3090 24G）用 uni-agent + verl + vLLM 0.18，跑通 SWE-Bench Verified 全量 500 条多并发推理（Qwen3-4B-Instruct-2507），8 卡打满，不开 llm-router。

## 最终结果

| 指标 | 值 |
|---|---|
| 跑批规模 | 500 条，零崩溃 |
| 耗时 | 3.5h |
| **Mean RM Score** | **0.1780** |
| resolved | 89/504 (17.66%) |
| ci_test | 122 passed (unit 113 + polling 4 + kv-event 5) |

## 定稿配置

| 项 | 值 | 原因 |
|---|---|---|
| 拓扑 | `nnodes=1 n_gpus=8` | 单机不能配多节点，否则 verl 跨节点 barrier 永等（死锁根因） |
| 并行 | `tp=2 → dp=4` | 8 卡 4 副本 |
| concurrency | **16** | uvloop `bad-fd` SIGABRT 的安全阈值（32/64 都崩，8 稳但慢） |
| CAP | **70000** | 99328 触发 vLLM HTTP server 死锁，妥协换出分 |
| 显存 | `gpu_mem=0.9`、`max_num_seqs=64` | 3090 从默认 1024 降下来避 sampler warmup OOM |
| uvloop | 保留 | verl 异步路径强依赖，关掉必锁 |

## 关键根因定位

1. **拓扑误配**：`nnodes=4` → 改 `nnodes=1`，崩溃+死锁全部消失。
2. **Ray actor idle timeout**：默认 10s idle 提前 `INTENDED_SYSTEM_EXIT`（只跑 4-8 条）→ `idle_worker_killing_time_threshold_ms=999999999` 根治。
3. **swe-rex 沙箱 hang**（py-spy 实锤）：AgentLoopWorker 卡在 `subprocess.communicate` 永久等待，与 verl/vLLM/Ray 无关。
4. **uvloop bad-fd**（concurrency≥32）：verl `auto_await` 临时线程 `asyncio.run` 与 uvloop 跨线程竞态，只能降并发规避。
5. **GPU 残留进程**：`VLLM::Worker`（带 `::`）占满 23GiB/卡 → 宿主层精确 pkill 清理。

## 工程产出（已 commit/push）

- `scripts/infer_multi.sh`（多并发入口，保留 `infer.sh` 单并发）
- `scripts/pull_swebench_images.sh` + README Step 4.5/5.5
- `examples/agent_interaction/parallel_infer.py` 增 `--max-num-seqs` / `--max-model-len` 参数、router_config_path None guard
- 离线 swe-rex wheels 挂载（`extra_run_args` + `--find-links`）
- Ray idle timeout 修复（`ray.init` _system_config）
- commit 7 → squash 成 2（infer + scripts），已 push

## git 状态

前三阶段代码已 commit/push 到 `main`，稳定。阶段四基于此主线推进。

## 备注

- resolved 89/504（17.66%）是 Qwen3-4B 模型能力上限，非工程问题（150 机 max_samples=4 验证 Mean RM 0.25，证明配置+模型能解题）。
- 仅 1/2016 轨迹撞 70K 截断。
