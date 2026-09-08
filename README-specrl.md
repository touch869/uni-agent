# SpecRL 实验运行

SpecRL 用当前 policy 验证旧 rollout 的 draft tokens，保留通过验证的连续前缀，减少逐 token decode。方法和代表结果见 [SpecRL 与 Toolcall](../specrl_toolcall_methods.md)。

## 环境与代码

使用已经配置好 PyTorch、Megatron、vLLM、Ray 和 uni-agent 依赖的 `uniagent-train-8` 容器，模型和数据位于共享的 `workspace`。实验脚本会清理容器内的 Ray 和 vLLM 进程，运行时应独占该实验容器。

SpecRL 使用 `specrl` 分支。当前主工作目录用于 Toolcall，首次准备时在宿主机创建独立 worktree，并初始化该分支对应的 verl：

```bash
git -C workspace/uni-agent worktree add --detach \
  workspace/uni-agent-specrl specrl
git -C workspace/uni-agent-specrl submodule update --init --recursive
```

已有该 worktree 时直接使用。子模块初始化可能需要访问代码仓库。下面的命令不切换现有 Toolcall 工作目录。

进入容器，设置代码目录。若容器已配置 OpenYuanRong 连接信息，保留现有值；否则按提示输入 address 和 key，key 不回显，也不写入文档：

```bash
docker exec -it uniagent-train-8 bash
export REPO_ROOT=workspace/uni-agent-specrl
cd "$REPO_ROOT"
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-${OPENYUANRONG_SERVER_ADDRESS:-}}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-${OPENYUANRONG_TOKEN:-}}"
if [ -z "$AKERNEL_SERVER_ADDRESS" ]; then
  read -r -p 'OpenYuanRong address: ' AKERNEL_SERVER_ADDRESS
fi
if [ -z "$AKERNEL_TOKEN" ]; then
  read -r -s -p 'OpenYuanRong key: ' AKERNEL_TOKEN
  printf '\n'
fi
export AKERNEL_SERVER_ADDRESS AKERNEL_TOKEN
```

## 短程 Demo

以下命令在容器内执行。使用 GPU 4–7，其中 2 张用于 Trainer、2 张用于 Rollout，两边 TP 均为 2。运行一组 Baseline/SpecRL 配对，共 6 steps，前 1 step 预热。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=workspace/models/Qwen3-1.7B \
REPEATS=1 TOTAL_TRAINING_STEPS=6 TOTAL_EPOCHS=6 METRIC_WARMUP_STEPS=1 \
TRAIN_NGPUS_PER_NODE=2 ROLLOUT_NGPUS_PER_NODE=2 TRAIN_TP=2 GEN_TP=2 \
TRAIN_MAX_SAMPLES=1 N=4 AGENT_MAX_TURNS=1 \
PROMPT_LENGTH=16384 RESPONSE_LENGTH=2048 SPECRL_BIAS=0.5 \
IGNORE_EOS=true FULL_DETERMINISM=true ASYNC_SCHEDULING=false \
PYTHONUNBUFFERED=1 \
bash examples/blackbox_recipes/mini_swe_agent/benchmark_specrl_controlled.sh
```

这是固定输出长度的性能实验，配置为单轮 Agent 交互；`6 steps` 指训练步骤。`N=4` 对应每个 prompt 的 4 条 rollout，也是本配置的 GRPO group size。Trainer 仍执行训练循环与权重同步；归档实验的 reward、loss、grad 为零，属于零有效 policy 更新的性能负载。

## 3090 / 1.7B 正式配对

同一容器中运行以下命令，使用 8 张 GPU，Trainer 和 Rollout 各 4 张。每组运行 30 steps，前 5 steps 预热，共 5 组配对，执行顺序由脚本交替安排。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=workspace/models/Qwen3-1.7B \
REPEATS=5 TOTAL_TRAINING_STEPS=30 TOTAL_EPOCHS=30 METRIC_WARMUP_STEPS=5 \
TRAIN_NGPUS_PER_NODE=4 ROLLOUT_NGPUS_PER_NODE=4 TRAIN_TP=2 GEN_TP=2 \
TRAIN_MAX_SAMPLES=1 N=4 AGENT_MAX_TURNS=1 \
PROMPT_LENGTH=16384 RESPONSE_LENGTH=2048 SPECRL_BIAS=0.5 \
IGNORE_EOS=true FULL_DETERMINISM=true ASYNC_SCHEDULING=false \
PYTHONUNBUFFERED=1 \
bash examples/blackbox_recipes/mini_swe_agent/benchmark_specrl_controlled.sh
```

默认训练集为 `workspace/data/swe_agent/swe_rebench_filtered_openyuanrong.parquet`，验证集为 `workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet`；通过 `TRAIN_DATA`、`VAL_DATA` 覆盖。上面使用当前 `specrl` 分支的 3090 脚本；A100/8B 的代表结果及配置口径见方法文档 2.4 节。

## 输出与查看

默认输出到 `$REPO_ROOT/outputs/specrl_benchmark/specrl_controlled_<时间戳>/`，也可在启动命令中设置绝对路径 `BENCHMARK_DIR`。每个 `repeat_XX_baseline/`、`repeat_XX_specrl/` 包含：

- `console.log`：启动配置、训练进度和控制台日志。
- `metrics.jsonl`：逐训练 step 的计时及 SpecRL 指标。
- `wall.json`：该次运行的端到端时间和退出码。

脚本结束时自动执行分析，生成 `report.md`、`summary.json` 和 `analyzer.log`。重新分析已有结果，在容器内执行：

```bash
python3 examples/blackbox_recipes/mini_swe_agent/analyze_specrl_benchmark.py \
  demo/experiments/specrl_demo_gpu4567 --warmup-steps 1
```

该命令会更新指定目录中的分析报告。正式 30-step 实验使用 `--warmup-steps 5`。重点查看 Rollout、Trainer 和端到端加速比，以及 draft 接受率、跨版本验证次数和复用 token 数。当前实验 `bias=0.5`，接受概率为 `min(1, exp(0.5)*p_i/q_i)`。
