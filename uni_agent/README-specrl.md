# Toolcall 实验运行

Toolcall 命中历史 observation 后，提前生成下一轮候选，同时真实执行工具；工具消息和上下文校验通过后复用候选。方法和代表结果见 [SpecRL 与 Toolcall](../specrl_toolcall_methods.md)。

## 环境与连接

使用已经配置好依赖的 `uniagent-train-8` 容器，代码目录为 `workspace/uni-agent`，分支为 `toolcall`。以下实验使用 GPU 4–7，TP=4；运行前确认这些卡可用。

从宿主机进入容器，然后配置 OpenYuanRong address 和 key。已有环境变量会直接沿用；未设置时交互输入，key 不回显：

```bash
docker exec -it uniagent-train-8 bash
cd workspace/uni-agent
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

配置入口是 `examples/agent_interaction/agent_config_openyuanrong.yaml`。对比脚本自动把当前代码及其 verl 加入 `PYTHONPATH`。

## SWE-bench：一组 100-turn 对照

以下命令均在容器内执行，一次运行包含 Baseline 和 Toolcall 两种模式：

```bash
GPU_IDS=4,5,6,7 N_GPUS_PER_NODE=4 TENSOR_PARALLEL_SIZE=4 \
MODEL_PATH=workspace/models/SWE-Lego-Qwen3-8B \
DATA_PATH=workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet \
INSTANCE_ID=astropy__astropy-12907 \
MAX_SAMPLES=1 NUM_WORKERS=1 MAX_TURNS=100 REPEATS=1 \
CONTROLLED_TOOL_BENCHMARK=0 \
bash workspace/compare_toolcall_sample1.sh
```

`MAX_TURNS=100` 是交互轮数上限，Agent 提交完成后可以提前结束。脚本默认 `TEMPERATURE=0`、`TOP_P=1.0`、`SEED=42`，开启确定性运行。重复统计时改为 `REPEATS=3` 或 `5`，脚本交替安排两种模式的执行顺序。

## 0.5 秒固定工具对照

```bash
GPU_IDS=4,5,6,7 N_GPUS_PER_NODE=4 TENSOR_PARALLEL_SIZE=4 \
MODEL_PATH=workspace/models/SWE-Lego-Qwen3-8B \
DATA_PATH=workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet \
INSTANCE_ID=astropy__astropy-12907 \
MAX_SAMPLES=1 NUM_WORKERS=1 MAX_TURNS=100 REPEATS=3 \
CONTROLLED_TOOL_BENCHMARK=1 CONTROLLED_TOOL_DELAY=0.5 CONTROLLED_TOOL_REPEATS=10 \
bash workspace/compare_toolcall_sample1.sh
```

受控模式仍通过数据集初始化环境，但将任务替换为固定工具调用：要求执行 10 次下面的命令，然后提交。

```bash
sleep 0.5 && printf 'TOOLCALL_BENCH_OK\n'
```

该模式跳过 reward evaluation，用于观察工具等待和模型生成的重叠收益。`CONTROLLED_TOOL_REPEATS` 是要求的调用数，报告中的 `calls(base/tool)` 是实际调用数；完整工作量按两者一致核对。文档 3.5.1 的归档结果要求 10 次、实际均完成 8 次，1.303× 对应双方匹配的实际轨迹耗时。新运行以当次报告的实际次数及计时范围为准。

## 结果与日志

默认输出目录为 `workspace/toolcall_comparisons/<时间戳>/`，启动时可设置 `OUTPUT_DIR` 指定新目录。

- `comparison.txt`、`comparison.json`：配对结果、加速比及可比性检查。
- `baseline_1.console.log`、`toolcall_1.console.log`：控制台输出。
- `baseline_1.json`、`toolcall_1.json`：推理结果与计时。
- `baseline_1.samples/`、`toolcall_1.samples/`：逐样本 interaction result 与运行日志。
- `runtime_environment.txt`：数据、GPU、受控工具等配置。

脚本启用 `PYTHONUNBUFFERED=1` 并通过 `tee` 实时输出；逐轮详细信息查看样本日志和 interaction result，控制台不固定每轮打印一行。只重新汇总已有结果时执行：

```bash
REPORT_ONLY=1 \
OUTPUT_DIR=demo/experiments/toolcall_controlled_0_5s \
REPEATS=3 CONTROLLED_TOOL_BENCHMARK=1 \
CONTROLLED_TOOL_DELAY=0.5 CONTROLLED_TOOL_REPEATS=10 \
bash workspace/compare_toolcall_sample1.sh
```

该命令会更新指定目录中的 comparison 报告，不调用模型。需要接着完成一次中断的实验时，使用原参数、原 `OUTPUT_DIR` 加 `RESUME=1`，脚本复用已成功完成的 case。

阅读结果时区分 `wall_speedup`（含启动）、`generation_speedup`（交互阶段）和 `prefix_wall_speedup`（匹配轨迹步骤）。候选重点看 started/committed/aborted 和 saved overlap；同时核对 action、observation、token 和实际工作量是否一致。
