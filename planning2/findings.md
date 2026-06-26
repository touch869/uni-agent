# Findings & Decisions

## Requirements
<!-- 从用户请求捕获 -->
- 4 台服务器做 swebench 沙箱,每台并发 32(总 128),避开单机 128 崩溃
- per-trajectory 起停容器(非常驻池,因 500 unique image × n=4=2000,常驻不现实)
- worker 用 asyncssh 远程 docker(用户拍板)
- 4 台各预 pull 全集 image(用户拍板)
- 参考 `uni_agent/deployment/vefaas/deployment.py` 实现"局域网 remote docker 环境"
- 代码改动要小、热插拔(yaml `type: remote_docker` 切换)

## Research Findings

### 1. 崩溃根因(非 docker 资源满)— 逐项实测排除
| docker/系统资源 | 147 实测值 | 结论 |
|---|---|---|
| 磁盘(overlay2) | 3.6T 用 18% | ✗ |
| 内存 | 251G 总/6G 用/245G 空闲 | ✗ 非 OOM |
| fd(ulimit) | 1048576,实际 3616 | ✗ |
| epoll max_user_watches | 5400 万 | ✗ |
| kernel.pid_max | 65536,峰值预估 ~11000 | ✗ |
| docker daemon | journalctl 无 error | ✗ |
| vm.max_map_count | 65530(接近但非瓶颈) | ✗ |

**真根因**:uvloop epoll fd 生命周期竞态,放大自 `auto_await` 多线程模型。
- 崩溃栈:`verl/agent_loop.py:1112 asyncio.gather` → `verl/utils/ray_utils.py:135 ThreadPoolExecutor+asyncio.run`(auto_await Case 3:新线程+新 event loop)→ `uvloop.loop` `uv__epoll_ctl_prep.cold` → abort(SIGABRT,不抛异常)
- 机制:Ray worker 已在 loop 里时,auto_await 每次调用起新线程+`asyncio.run()` 创建全新 event loop;worker 内 `agent_loop.py:556-560` 用 gather 并发跑 `CONCURRENCY//NWORKERS` 条 trajectory(128//8=16),每 trajectory 起 docker 容器+swerex+子进程,fd 全在该 loop watch;uvloop 在 fd 高频创建/关闭+多线程交错时 `epoll_ctl` 拿到已关闭却仍 watch 的 fd → EBADF/EPERM → libuv 直接 abort。
- uvloop 0.22.1,非默认 policy(`_UnixDefaultEventLoopPolicy`),ray 源码无 uvloop——某处显式 `set_event_loop_policy`,未定位。
- **崩点边界**:CONCURRENCY=16(0624)稳定跑完 256 rollout ✓;CONCURRENCY=128+16样本探针没崩(样本少在途没到 128)✓;CONCURRENCY=128+全集 ~98 容器崩 ✗。崩点在 16~128 间,未二分。

### 2. 部署架构(可复用的关键设计)
- `AbstractDeployment`(swerex)接口:`start/stop/is_alive/runtime`,所有 deployment 实现它
- `RemoteRuntime`(`uni_agent/deployment/remote_runtime.py`)= **纯 HTTP 客户端**(aiohttp),通过 `/is_alive`/`/create_session`/`/run_in_session` 远程操作 swerex server,**本地不起子进程、不 watch fd**。这是避开 uvloop fd 竞态的银弹
- `RemoteRuntimeConfig` 已支持 `host:port` 直连模式(`remote_runtime.py:104-111`),不一定要 veFaaS 的 base_url
- vefaas 模式:云沙箱,`create_sandbox`(火山 SDK)起远端容器 → `RemoteRuntimeConfig(base_url=function_route)` HTTP 连。**我们要做的就是把 create_sandbox 换成"SSH 远端 docker run"**
- `LocalDeployment`(`local/deployment.py`)已是"docker run subprocess + RemoteRuntime HTTP 连"混合模式,`_start_oci_container:278-291`:`docker run`(本地子进程)→ 拿容器 IP → `RemoteRuntimeConfig(host/port)`。**崩点就在这个本地 subprocess docker run**——换成 SSH 远端即解

### 3. image 注入路径(关键利好:不改 dataset/上游)
- yaml `agent_config_localdocker.yaml` 的 deployment **无 image 字段**(command 是 pip install swe-rex + swerex.server)
- image 由 dataset 的 `tools_kwargs.env.deployment.image` per-trajectory 注入
- 链路:`agent_loop.py:270-277` `_deep_merge(yaml, tools_kwargs)` → `env.deployment.image` 被 override → `AgentEnvConfig` → `env_config.deployment.get_deployment(run_id)`
- **实测 dataset**:500 行,500 unique image,每条 `image: swebench/sweb.eval.x86_64.<proj>_1776_<num>`,`type` 不在 dataset(由 yaml 提供)
- `_deep_merge`(`agent_loop.py:27-42`):dict 深合并,override wins。新 config 声明 `image` 字段即自动拿到 per-instance image,**免费**

### 4. config 注册体系(热插拔)
- `DeployConfig = Annotated[Vefaas|Local|LocalAttach|Host|LocalNative|Modal, Field(discriminator="type")]`(`config.py:184-192`)— 联合类型,type 字段判别
- `_LAZY_EXPORTS`(`__init__.py:13-20`)— 类名→模块路径,`__getattr__` 懒加载
- 每个 Config 有 `get_deployment(run_id)` 工厂方法(`env.py:62` 调用)
- 加新 deployment:① `config.py` 加 Config 类 + 加入联合类型 ② `__init__.py` 加 `_LAZY_EXPORTS` + `__all__`

### 5. image/wheels 现状(147)
- 147 已有 **1001 个 swebench image**(全集 + 更多,已 pull 过),不会触发 pull
- 另 3 台机器可能无 image → 首次跑触发 pull(每 image 几 GB,500 个海量),靠预 pull 脚本规避
- local yaml command 依赖挂载 `/data1/hgq/swe_wheels:/wheels:ro` 装 swe-rex,远端机需有 swe_wheels 目录

### 6. SSH 环境(147→4 台)
- 147 `root@8.92.9.147` 免密登录已确认(本会话验证)
- 147 有 `/root/.ssh/id_rsa.pub`,可推到 4 台 authorized_keys 实现 asyncssh 免密
- asyncssh/paramiko **均未装**(147 vllm021 容器内),需 `pip install asyncssh`

## Technical Decisions
| 决策 | 选择 | 理由 | 备选(为何不选) |
|---|---|---|---|
| 沙箱生命周期 | per-trajectory SSH docker run/rm | 2000 image 常驻不现实,隔离干净 | local_attach 常驻池:500/台常驻 + 环境污染 |
| SSH 客户端 | asyncssh | 纯 asyncio 零子进程 fd,契合消除 uvloop fd 压力 | subprocess ssh:每 trajectory spawn 子进程,reintroduce fd |
| image 分发 | 4 台各预 pull 全集 | per-trajectory 不 pull、快,适合多次跑 | lazy pull:首跑极慢;147 中转:多一套 |
| 负载分配 | hash(run_id)%len(hosts) 无中央调度 | 天然分散幂等,无单点 | 中央调度服务:额外组件+单点 |
| runtime | 直接复用 RemoteRuntime | 纯 HTTP,host 填远端 IP 即可,零改 | 重写 runtime:无谓 |
| 部署注册 | 加 type=remote_docker 进联合类型 | 热插拔,yaml 一行切换 | 改 local:破坏现有 |

## 待验证/风险
- asyncssh 在 Ray auto_await 多线程 event loop 里是否稳(理论比 uvloop+subprocess 强,未实证)
- 4 机 asyncssh 短连接数:128 并发起停=高频 SSH 握手,可能成新 fd 瓶颈 → 首版若崩改连接池
- 远端机 docker daemon 并发 32 容器是否稳(147 单机 98 容器是 uvloop 崩非 docker 崩,32 应安全)
- 4 台磁盘:500 image × 几 GB ≈ 1-2TB/台,147 /data1 3.6T 用 18% 够,其他台待确认
- uvloop 安装点未定位(若关 uvloop 是另一条修复路径,本任务走分散沙箱而非关 uvloop)
