# 运行环境说明

本文记录训练 / 推理使用的 GPU 服务器。当前两台：

- **8.92.9.150** —— 共享多租户，只用 GPU 6、7。
- **8.92.9.147** —— 独占（当前 `infer.sh` 用 2 卡，后续可扩到 8）。

> 采集时间：2026-06-17（`nvidia-smi` / `docker` / `cat` 实测）。

---

## 8.92.9.150（共享多租户）

### 连接

```
ssh root@8.92.9.150
```

- 主机名：`mindspore`
- 系统：Ubuntu 20.04.6 LTS（内核 5.4.0-216-generic）

> 该机为**多租户共享**：同时运行着多人的 verl / swebench 容器（`swe-lmytest`、`yrf-swe`、`zzq_rl`、`lll-rl-*` 等）。我们只用其中两张卡（见下）。

### 硬件

| 项 | 规格 |
|---|---|
| CPU | Intel Xeon Gold 6226R @ 2.90GHz，64 核 |
| 内存 | 251 GiB |
| GPU | **8 × NVIDIA GeForce RTX 3090**，每张 24 GiB |
| 驱动 | 575.57.08 |
| GPU 算力 | compute capability 8.6 |
| 数据盘 | `/data1` = 3.6 TB，约 768 GB 可用（已用 78%） |

GPU 编号 0–7。CUDA 工具链以容器内 torch 自带的为准，宿主 `nvcc` 不在 PATH。

### 容器 `hgq-swe`

| 项 | 值 |
|---|---|
| 镜像 | `verlai/verl:vllm018.dev1` |
| 状态 | Up 5 days |
| runtime | `runc`（非 nvidia runtime；但镜像内置 NVIDIA 驱动库，容器内 `nvidia-smi` 可见全部 8 张卡） |
| 挂载 | `/data1 → /data1`、`/tmp → /tmp`、`/var/run/docker.sock`（Docker-in-Docker）、`/usr/bin/docker` |

> 挂了 docker.sock，所以 `hgq-swe` 本身常作为**编排容器**，可在内部再起 GPU 任务容器。

容器内软件栈（`docker exec` 实测）：

| 包 | 版本 |
|---|---|
| Python | 3.12.3 |
| torch | 2.10.0+cu129（CUDA 12.9） |
| vLLM | 0.18.0 |
| Ray | 2.54.1 |
| verl | **镜像内未安装**，从仓库的 `verl/` 子模块以 editable 方式使用 |

### 工作目录与 GPU 占用

仓库部署在 `/data1/hgq/uni-agent`。`env.sh` 内容：

```bash
#export HF_ENDPOINT=https://hf-mirror.com
alias gpu='nvidia-smi'
export CUDA_VISIBLE_DEVICES=6,7
```

**我们固定使用 GPU 6、7**（两张 3090，共 48 GiB）。进容器后先 `source /data1/hgq/env.sh` 再启动任务。

### 操作约束（硬性安全边界）

> 对该机器的任何自动化操作，必须先满足以下三条：

1. **禁止越界写文件**：绝对禁止删除或改写 `/data1/hgq` 目录**以外**的任何文件。
2. **禁止越界动容器**：绝对禁止操作或改写 `hgq-swe` **以外**的任何 docker 容器（多租户共享机，他人容器一律不碰）。
3. **高危操作需审批**：即便在 `/data1/hgq` 内或 `hgq-swe` 容器内，`rm`、`docker rm`、批量覆盖等高危命令前，**必须先经用户审批**。

### 代码同步流程

1. **WSL（本地仓库 `uni-agent2`）**：改完 → `git commit` → `git push`
2. **GPU 机（`/data1/hgq/uni-agent`）**：`git pull --rebase`

> 两端是同一仓库的不同 checkout，用 `--rebase` 保持线性历史。

---

## 8.92.9.147（独占）

### 连接

```
ssh root@8.92.9.147
```

- 主机名：`mindspore`
- 系统：Ubuntu 18.04.6 LTS（内核 4.15.0-193-generic）
- **独占**，无其他用户 / 容器。

### 硬件

| 项 | 规格 |
|---|---|
| CPU | Intel Xeon Gold 6226R @ 2.90GHz，64 核 |
| 内存 | 251 GiB（可用约 247 GiB） |
| GPU | **8 × NVIDIA GeForce RTX 3090**，每张 24 GiB |
| 驱动 | 575.64.05 |
| GPU 算力 | compute capability 8.6 |
| 数据盘 | `/data1` = 3.6 TB，约 3.2 TB 可用（已用 8%） |
| 系统盘 | `/` = 431 GB，约 75 GB 可用（已用 82%） |

> `/data1` 基本空的，符合独占 / 刚启用。

### 容器 `hgq-swe`

| 项 | 值 |
|---|---|
| 镜像 | `verlai/verl:vllm018.dev1`（与 8.92.9.150 **同镜像**） |
| 状态 | Up 5 hours |

软件栈与 8.92.9.150 一致（py3.12 / torch 2.10+cu129 / vLLM 0.18 / Ray 2.54；verl 走子模块）。

### 工作目录与 GPU 占用

仓库在 `/data1/hgq/uni-agent`。`/data1/hgq` 下另有 `agentic_log/`、`swe/`、`env.sh`，以及装机残留 `NVIDIA-Linux-x86_64-575.64.05.run`、`vllm.tar`。

`env.sh` 是 `CUDA_VISIBLE_DEVICES=6,7`，`infer.sh` 当前也用 2 卡（6,7），**与 .150 一致**。本机独占，后续有需要可扩到全部 8 卡（把 `CUDA_VISIBLE_DEVICES` 改为 `0,1,2,3,4,5,6,7` 或注释掉），暂不改。

> 操作约束与 .150 完全一致（见下）——独占只意味着将来可用满 8 卡，不代表放宽写文件 / 动容器 / 高危操作的边界。

### 操作约束（硬性安全边界）

> 本机虽为独占，但为限制 agent 的操作权限，**沿用与 8.92.9.150 完全相同的硬性约束**。任何自动化操作前必须满足：

1. **禁止越界写文件**：绝对禁止删除或改写 `/data1/hgq` 目录**以外**的任何文件。
2. **禁止越界动容器**：绝对禁止操作或改写 `hgq-swe` **以外**的任何 docker 容器。
3. **高危操作需审批**：即便在 `/data1/hgq` 内或 `hgq-swe` 容器内，`rm`、`docker rm`、批量覆盖等高危命令前，**必须先经用户审批**。

### 代码同步流程

同 8.92.9.150：WSL `commit` / `push` → 本机 `/data1/hgq/uni-agent` `git pull --rebase`。
