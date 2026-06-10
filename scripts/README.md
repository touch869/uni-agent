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