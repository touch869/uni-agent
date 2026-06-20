# Scripts

按以下顺序执行：

## 1. 克隆项目

```bash
git clone https://github.com/AssassinGQ/uni-agent.git
```

## 2. 创建 Docker 容器

```bash
cd uni-agent
CONTAINER_NAME=swe-xxx bash scripts/create_docker.sh
cd -
```

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONTAINER_NAME` | `hgq-swe` | 容器名称 |
| `IMAGE_NAME` | `verlai/verl:vllm018.dev1` | 镜像名称 |
| `SHM_SIZE` | `10g` | 共享内存大小 |

## 3. 进入 Docker 容器

```
docker exec -it swe-xxx bash
cd /path/to/uni-agent
```

## 3. 安装依赖

```bash
bash scripts/install-uni-agent.sh
```

初始化 git submodule 并安装 verl 及相关依赖。

## 4. 准备数据集

```bash
# 使用默认值 (modal)
bash scripts/prepare_dataset.sh

# 指定部署后端
bash scripts/prepare_dataset.sh vefaas
```

可选参数：

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `DEPLOYMENT` | 第 1 个 | `modal` | 数据集镜像类型 |

DEPLOYMENT 决定写入 parquet 的沙箱镜像名：

| 值 | 生成的镜像名 | 适用场景 |
|------|------|------|
| `modal` | `swebench/sweb.eval.x86_64.*` | local docker / modal 部署 |
| `vefaas` | 阿里云 veFaaS 镜像 | 仅 veFaaS 部署 |
| `local` | 尚未实现 | — |

> **使用 local docker 沙箱时，`DEPLOYMENT=modal` 即可**（默认值），docker 会自动拉取 `swebench/sweb.eval.x86_64.*` 镜像。

输出文件：`scripts/swe_bench_verified_<deployment>.parquet`

## 4.5 预下载 swe-rex wheels + 拉取 SWE-bench 镜像（首次必做）

多并发推理时，每个沙箱都要 `pip install swe-rex`。500 个沙箱并发 pip 会把清华源打满（超时/卡死），且 Docker Hub 拉镜像国内超时。此脚本一次性解决两个问题：

```bash
bash scripts/pull_swebench_images.sh
# 或指定数据集路径
bash scripts/pull_swebench_images.sh scripts/swe_bench_verified_modal.parquet
```

脚本做两件事：
1. **预下载 swe-rex + 全部依赖 wheels** 到 `/data1/hgq/swe_wheels`（26 个，5.5MB），沙箱挂载后 offline pip install（秒级，不走网络）。agent_config 已配置 `--find-links /wheels` + 清华 fallback（兼容 pydantic-core 等平台相关包）。
2. **从火山引擎 CR 拉取 SWE-bench 镜像**（`enterprise-public-cn-beijing.cr.volces.com`，国内快），`docker tag` 成 `swebench/sweb.eval.x86_64.*:latest`（parquet 期望的格式）。跳过已有的镜像。

> 不跑此脚本直接推理，会导致：pip install 慢/超时 + 镜像 pull Docker Hub 超时 → 沙箱起不来 → 大量样本 fail。

## 5. 运行推理

```bash
# 使用默认值
bash scripts/infer.sh

# 只指定模型路径（建议配置）
bash scripts/infer.sh /data/models/Qwen3-4B

# 指定模型和数据集路径
bash scripts/infer.sh /data/models/Qwen3-4B /data/dataset.parquet

# 指定全部参数
bash scripts/infer.sh /data/models/Qwen3-4B /data/dataset.parquet my_config.yaml
```

可选参数：

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_PATH` | 第 1 个 | `/path/to/Qwen3-4B` | 模型路径（建议配置） |
| `DATA_PATH` | 第 2 个 | `scripts/swe_bench_verified_modal.parquet` | 数据集路径 |
| `AGENT_CONFIG` | 第 3 个 | `scripts/agent_config_localdocker.yaml` | Agent 配置路径 |

> DATA_PATH 和 AGENT_CONFIG 一般不用配置，使用默认值即可

脚本会在启动前检查以上三个路径是否存在，缺失则报错退出。

## 5.5 多并发推理（全量 500 条，8 卡 data-parallel）

`infer.sh` 是单并发 smoke test（2 卡、1 样本）。要跑全量 500 条 + 多并发，用 `scripts/infer_multi.sh`：

```bash
# 最简：只需指定模型路径
bash scripts/infer_multi.sh /data/models/Qwen3-4B

# 指定模型 + 数据集
bash scripts/infer_multi.sh /data/models/Qwen3-4B /data/dataset.parquet

# 通过环境变量调参（可选）
MAX_SAMPLES=-1 TP=2 NWORKERS=8 MAX_NUM_SEQS=64 \
  bash scripts/infer_multi.sh /data/models/Qwen3-4B
```

默认参数（适配 8×3090 / Qwen3-4B）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/path/to/Qwen3-4B` | 模型路径（第 1 个参数） |
| `DATA_PATH` | `scripts/swe_bench_verified_modal.parquet` | 数据集路径（第 2 个参数） |
| `AGENT_CONFIG` | `scripts/agent_config_localdocker.yaml` | Agent 配置（第 3 个参数） |
| `NNODES` | `1` | 物理节点数（单机用 1，**不是 GPU 数**） |
| `NGPUS` | `8` | 每节点 GPU 数 |
| `TP` | `2` | tensor parallel；dp = NNODES × NGPUS / TP |
| `NWORKERS` | `8` | agent rollout worker 数 |
| `MAX_NUM_SEQS` | `64` | 每实例 vLLM 并发序列（24GB 卡用 64） |
| `MAX_SAMPLES` | `-1` | 样本数（-1 = 全量 500） |
| `MAX_TURNS` | `100` | 每样本最大交互轮数 |
| `PROMPT_LEN` | `32768` | prompt 长度 |
| `RESPONSE_LEN` | `65536` | response 长度 |

> **前置条件**：必须先跑过 Step 4.5（预下载 wheels + 拉镜像），否则沙箱起不来。
>
> **日志重定向**：`infer_multi.sh` 输出到 stdout/stderr，自行重定向即可：
> ```bash
> setsid nohup bash scripts/infer_multi.sh /data/models/Qwen3-4B > logs/run.log 2>&1 &
> ```

## 6. 已知问题：transformers / numpy 异常

在 Docker / 内核 / 系统版本较旧的环境（如 Docker 20.10、内核 4.15、Ubuntu 18.04）下，同一镜像会触发以下两个运行时异常，导致推理起不来；在较新环境（如 Docker 26.x、内核 5.4、Ubuntu 20.04）上开箱即用，可跳过本节。根因是宿主运行时环境差异，非包/版本本身。

| 现象 | 根因 | 解决方案（容器内执行一次） |
|------|------|--------------------------|
| `import transformers` 报 `Backend should be defined in the BACKENDS_MAPPING. Offending backend: tf` | 旧环境下 transformers 5.3.0 backend 检查异常 | `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers==4.57.6"` |
| vllm worker 起不来，`RecursionError`（numpy `issubdtype`↔`__repr__` 递归，根源 `np.dtype(bfloat16)`） | 旧环境下 numpy 混入的 2.x overlay 触发 dtype repr 死循环 | `pip uninstall -y numpy && rm -rf /usr/local/lib/python3.12/dist-packages/numpy /usr/local/lib/python3.12/dist-packages/numpy-1.26.4.dist-info && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy==1.26.4"` |

> `transformers==4.57.6` 会顺带降级 `huggingface_hub`→0.36.2；numpy 重装后 `pip check` 报 `opencv-python-headless requires numpy>=2` 可忽略。`rm` 类命令在共享机上执行前先审批。