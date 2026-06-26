# Progress Log

## Session: 2026-06-25

### Phase 0: 规划与根因调查
- **Status:** complete
- **Started:** 2026-06-25
- Actions taken:
  - 承接 8B/CONCURRENCY=128 全集崩溃调查,逐项排除 docker/系统资源(磁盘/内存/fd/epoll/pid/daemon 均非瓶颈)
  - 定位真根因:uvloop epoll fd 竞态,放大自 `verl/utils/ray_utils.py:135` auto_await 多线程新 event loop 模型;完整崩溃栈记录(记忆 c128-uvloop-abort-crash)
  - 用户提方案:4 台服务器分散沙箱,每台 32 并发
  - 勘探部署架构:`vefaas/deployment.py`(云沙箱模式)、`remote_runtime.py`(RemoteRuntime 纯 HTTP 客户端,可复用)、`local/deployment.py`(docker run+HTTP 连混合模式,崩点在本地 subprocess docker run)、`config.py`(DeployConfig 联合类型,热插拔)、`agent_loop.py:270-277`(image 经 _deep_merge 从 tools_kwargs 注入,免费)
  - 实测 dataset:500 行 500 unique image,每条 `image: swebench/sweb.eval.x86_64.<proj>_<num>`
  - 实测 147:已 1001 个 swebench image、root 免密 OK、asyncssh/paramiko 均未装
  - 用户拍板:asyncssh + 4 台各预 pull 全集
  - 用 AskUserQuestion 定 3 个关键决策(沙箱生命周期/SSH 客户端/image 分发)
  - 写 task_plan.md + findings.md + progress.md 到 planning2/
- Files created/modified:
  - planning2/task_plan.md(新建)
  - planning2/findings.md(新建)
  - planning2/progress.md(本文件)
  - `workhis/planning-history/20260625-remote-docker-deployment/task_plan.md`(早先写,内容与 planning2/ 一致,可作备份)
- Errors encountered:
  - 多次 ssh 命令因中文/括号经多层 shell 转义报 syntax error → 改纯 ASCII + 写脚本文件 scp 执行规避
  - ExitPlanMode 被用户拒(改走 planning-with-files 工作流,写 planning2/)
- Next:
  - 待用户审批 task_plan.md 后进 Phase 0:147 local-docker 32 并发基线(先验证档位不崩,再写 remote_docker)

### 计划变更:新增 Phase 0
- **Status:** complete(计划调整)
- 变更:用户要求增加 Phase 0——147 local-docker 跑通 32 并发,作为基线 + 验证单机 32 档位不崩
- 理由:① Phase 2 每台目标负载是 32,先在本机验证这个档不崩(单机 128 崩了,32 未实测);② 作 remote_docker 4×32 的对照基线
- task_plan.md 已插入 Phase 0(不写新代码,用现有 local yaml + CONCURRENCY=32),原 Phase 1/2/3 顺延不变

### Phase 0 执行:32 并发探针(2026-06-25 07:16-07:30)
- **Status:** complete(结论:32 不可用)
- 启动:`CONCURRENCY=32 MAX_NUM_SEQS=8 PROMPT_LEN=16384 RESPONSE_LEN=8192 MAX_SAMPLES=16`,local yaml,147 容器
- 观察:
  - vLLM 8 replica 起完,KV 5.52GiB/replica,40192 tok,"Maximum concurrency for 25600 tokens/request: 1.57x"
  - 进程**没崩**(crash check=0,无 SIGABRT,跟 128 的崩不同)
  - 但 generation throughput 长期 0.2-0.3 tok/s(极低),prompt throughput 1800-2700(高)
  - KV usage 59-63%(没顶死,但单条 prompt 涨到 24785 接近 max_model_len 25600)
- 诊断 trajectory run.log:`STEP 11→18,每步 Prompt Tokens 24659→24785(+21), Completion Tokens: 1` → 1-token 退化死循环
- 全量分析 32 条 trajectory:**~26 条退化**(last_completion=1,步数 30-100 撞 max_turns),仅 ~7 条正常 → **退化普遍,非个例**
- 关键对比:CONCURRENCY=16(0624)同 config 跑通 256 rollout;CONCURRENCY=32(本实验)普遍退化 → **32 是退化触发档**
- 机制:32/8dp=4 trajectory/replica > 物理极限 1.57x,KV/生成空间不足;非 uvloop 崩、非 docker 资源满
- kill 探针 + 清环境(8 个 VLLM::Worker 僵尸进程 kill -9 释放显存)
- 记忆:`concurrency32-1token-degradation`(32 触发退化,16 不触发,分散沙箱不解决退化)
- **计划影响**:Phase 2/3 的 4×32 假设被推翻——分散沙箱不解决单 replica 退化。需用户决策:改 4×16(总64)/8×16(总128)/或先修退化再 4×32。
- Files:无代码改动(Phase 0 不写代码),无 commit。

### 退化根因深挖 + no space left on device 分析(2026-06-25)
- **Status:** complete(诊断阶段)
- 用户挑战:"No function call found 之前靠调大 gpu-mem 解决"+"若是 RESPONSE_LEN 问题,16 为啥没事"——两点都 valid,推翻我之前 RESPONSE_LEN 单一根因假设
- **挖出真坑 `parallel_infer.py:69`**:`max_model_len = min(prompt+response+1024, 70000)`
  - 默认 PROMPT=32768/RESPONSE=65536 → max_model_len=**70000**,远超 KV 池 40192 → vllm 并发仅 0.57x(连1个请求都放不下)→ **这就是历史"调大 gpu-mem 解决 No function call found"的真根因**(扩显存=扩KV池)
  - 退化实验 PROMPT=16384/RESPONSE=8192 → max_model_len=25600,1.57x,比默认好但仍不够
- **物理极限**:单 replica KV 池 40192 / 实际 prompt ~24785 ≈ 1.62x 并发。32/8dp=4 > 1.62x 超极限;16/8dp=2 略超但有余量
- **16 vs 32 差异机制**(回答用户挑战):差异在 **preempt 余量**非静态 config。16 并发 2 traj/replica,一个涨长可 preempt 让另一个跑完;32 并发 4 traj 全挤,preempt 哪个都救不了→雪崩
- **preempt=0 死锁(待验证)**:Phase 0 观察退化期 preempt=0。推测:退化 seq 每步只生成 1 token→KV 涨极慢→永不触发 preempt 阈值→正反馈死锁。也可能 verl 启动 vllm 时 preemption_mode=NONE
- **no space left on device**:查清是 docker cp archive API 的 bind mount(flags 0x5000)瞬时 ENOSPC,非磁盘满(/data1 18%/inode 5%/mount 48 全正常),32 并发压测期偶发,重试不复现。与退化零因果。规避:用 docker exec 替代 docker cp
- 记忆:`max-model-len-70000-default`(新建,链 [[concurrency32-1token-degradation]])
- **计划影响**:退化是 per-replica 的 max_model_len/preempt 问题,**分散沙箱不解决**(4×32 每台仍退化)。应先单机调通 max_model_len + 验证 preempt,再定要不要 4 机
- **Next**:待用户确认是否跑"32 并发 + 降 max_model_len + 盯 preempt 计数器"对照实验——这是解开退化根因的关键一测

### 方向转向:tp=2 dp=4 增大 KV 池(2026-06-25,用户提议)
- **Status:** 待执行(用户拍板)
- **起因**:用户提"KV 池 40192 需调大,改 tp=2 dp=4 还是 8 卡,会不会就行"——直击要害
- **日志铁证**:c32 探针确实是 `tensor_parallel_size=1, dp=8, KV池=40192`。tp=1 每卡装完整 8B 权重 16GB,KV 被挤到 5.5GiB
- **tp=2 dp=4 KV 池估算**(~4.5x):
  - 每卡权重 16GB→8GB(分摊),释放 8GB/卡 给 KV
  - 每卡 KV 显存 5.5GiB→~13GB;replica(2卡)KV 池 40192→~180000 tok
  - 每 traj 可用 KV:tp1=10048 → tp2=~22500(**2.2x**)。退化时 prompt 24785,tp1 装不下退化,tp2 基本装下
- **临界点**:32并发/dp4=8 traj/replica,8×25600=204800 vs 池子~180000,略超13%。稳法:降 max_model_len 到24000 / 或28并发 / 或保持32靠 preempt 正常工作
- **风险**:NCCL all-reduce 开销(3090 PCIe);activation 挤占实际 KV 池可能150000-180000;mooncake 若开 router 需验([[mooncake-tp-not-deadlock]] 说不死锁)
- **对计划影响(重要)**:tp=2 是比 remote_docker 更釜底抽薪的修复——remote_docker 分散沙箱但不增 KV 池,治标。若 tp=2 单机能跑 32 不退化,4 机真正价值 = 4×tp2 每机32(不退化)→ 总128,既绕 uvloop 崩溃(单机≤32)又解退化
- **新 Phase 0.5(建议插入,优先于 Phase 1-3)**:`TP=2 CONCURRENCY=32 MAX_NUM_SEQS=8 MAX_SAMPLES=16` 跑 16 样本,盯 KV 池实际值 + Maximum concurrency(1.57x→~7x) + 退化率(目标<5/16)
- **Next**:待用户确认跑 tp=2 实验;若通过,task_plan 的 Phase 1-3(remote_docker)需重排为"4 机 × tp2"而非"分散沙箱避退化"

### Phase 0.5 执行:tp=2 dp=4 验证(2026-06-25 09:24)
- **Status:** complete(KV 假设证实,退化大幅改善但未消除)
- **docker 坑(插曲,非 tp 问题)**:tp2 首两次 RM Score=0,误以为是 worker/dp 不匹配。深挖 run.log 发现真因:`docker run` exit 125,`error creating overlay mount: no space left on device`。根因=docker daemon overlay2 graphdriver 状态损坏(stale mount + EINVAL unmount + ENOSPC create),c32 沙箱残留 + 835 image/3276 层累积。ext4 健康(3.6T空/2.44亿inode空/无quota),内核手动 mount OK,纯 docker driver 问题。**`systemctl restart docker` 修复,docker run 恢复**。教训:run.log 在 `/data1/hgq/agentic_log/<run_id>/run.log`(非 stdout),静默 catch(agent_loop.py:139 catch→_build_empty_agent_output reward=0)使 RM Score=0 看似"没跑"实为"沙箱起失败"
- **tp2 实测(修复 docker 后)**:
  - KV 池/replica:40192→**195008 tok(4.85x)**;Max concurrency:1.57x→**7.62x**
  - vllm throughput:**0.2 tok/s → 254 tok/s(1200x)**,vllm 从退化空转变正常生成
  - 退化 step 级:~92%→**46%**(826 sample,383 个 Completion=1)
  - 沙箱正常起(13 个在跑),GPU 8 卡 22.7GB/卡
- **结论:KV 池是退化主因——证实**。tp2 增大池子大幅改善。**但单机 32 仍触物理极限**:8 traj/replica × prompt 24785 = 198280 需求 vs 池 195008,每 traj 平均 24389 < 24785,临界,46% traj 仍退化
- **要完全消除退化**:① 降并发到 ~24(6 traj/replica,KV 充足);② 或降 max_model_len(按 [[max-model-len-70000-default]] 思路);③ 或解决 preempt=0 死锁让 vllm 排队
- **对 4 机计划影响**:tp2 是釜底抽薪(KV 池 4.85x),remote_docker 分散沙箱不增 KV(治标)。正确用法:4 机 × tp2,每机并发降到不退化档(~16-24)→ 总并发 64-96 不退化;或解决 preempt 后每机 32→ 总 128
- **Next**:等最终 RM Score + per-traj 退化数;据结果定单机目标并发(16/24/32)再规划 4 机

### Phase 0.5 最终结论(2026-06-25 09:51,修正中间乐观判断)
- **最终数据**:RM Score **0.0625**(16 traj 仅 1 通过);per-traj **degraded=14 normal=2 = 87.5% 退化**;耗时 1612s
- **中间结论错了**:中途报"46% 改善"是 throughput 254 时的 step 级快照(正常 traj 在跑),最终 trajectory 级 87.5%,跟 c32 的 81% 几乎一样
- **tp2 没解决退化**:KV 池 4.85x 但 dp 减半→traj/replica 翻倍(4→8),每 traj 可用 KV 仅 2.43x(24389 vs c32 的 10048),vs prompt 24785 临界,正反馈退化
- **物理极限定论**:**单机 8 卡 32 并发无论 tp1/tp2 都必然退化**(traj/replica 太多,每 traj KV 撑不住 agent 多轮 prompt 累积到 24785)。tp2 增池被 traj/replica 翻倍抵消
- **真根因修正**:退化不是 KV 池总量问题,是 **traj/replica × prompt 累积 vs KV 池 + 退化正反馈**(一步退化→prompt 涨→更挤→不可逆)
- **不退化的两条路**:① 每 traj KV >> prompt(需低 traj/replica = 低并发或高 dp 多机);② 打断正反馈(让 vllm preempt 排队,当前 preempt=0 死锁)
- **可行方案**:
  - 4 机 × tp2 × 16 = 总 64 不退化(16 已知安全,0624 验证)— 最稳基线
  - 解决 preempt 死锁(查 verl 启动 vllm 的 preemption_mode)→ 若能排队,4 机 × 32 = 128 不退化(慢但稳)— 唯一 4 机达 128 的路
  - 8 机 × 16 = 总 128 不退化(机器翻倍)
- **结论推翻 Phase 0.5 初判**:tp2 不是釜底抽薪(退化没降);分散沙箱/增 KV 都治标。**唯一治本=降 traj/replica(降并发/多机)或修 preempt**
- **Next**:待用户拍板——验证 preempt 死锁(查 preemption_mode)还是直接走 4 机 × 16 基线

### 真根因突破:max_model_len 撞墙(2026-06-25 10:05,推翻所有 KV 假设)
- **铁证(tp2 退化 traj run.log 序列)**:
  ```
  Prompt 7766 → Completion 8192  (模型推理发散,生成满 RESPONSE_LEN 截断)
  Prompt 15978 → Completion 8192 (再截断,prompt 暴涨 +8192)
  Prompt 24190 → Completion 386
  Prompt 24596 → Completion 1    (prompt 接近 max_model_len 25600,模型输出 1 token EOS)
  Prompt 24617→25583 → Completion 1 (死循环,每步 prompt +21,撞 max_model_len)
  ```
- **KV 池假设彻底推翻**:tp2 vllm 日志 `KV cache usage: 10-30%`,**KV 池大量空闲**(195008 只用 20%),prefix hit 96-98%。退化跟 KV 池大小无关。tp2 增池到 195008 退化率仍 87%(跟 c32 81% 一样)
- **preempt=0 不是死锁**:verl 不配 preemption_mode(grep 空),vllm 默认;preempt=0 因 KV 没满,不需要 preempt
- **真根因**:`parallel_infer.py:69` `max_model_len = PROMPT_LEN + RESPONSE_LEN + 1024 = 25600`,**只够 1 轮 response**。agent 多轮累积 + Qwen3 某步推理发散(生成满 8192)→ prompt 暴涨 → 撞 max_model_len → 模型输出 1 token EOS → 解析失败 → 死循环
- **用户最初质疑"输出长度配置不对"完全正确**(三次 KV 方向诊断都错了:KV 不够→preempt 死锁→tp2 增池)
- **修复**:KV 池大量空闲,可放心增大 max_model_len。改 parallel_infer.py:69 公式(如 `PROMPT_LEN + RESPONSE_LEN*3` = 40960),给 agent 多轮累积留空间。8 traj/replica × 40960 = 327680 vs KV 195008... 待验证(可能需同时降 RESPONSE_LEN 或并发)
- **正常 traj 对照**:max completion 6823(不撞 8192),prompt max 22250(远离 25600)→ 不退化
- **Next**:改 max_model_len 公式重跑 32 并发,验证退化消除

### 🎉 真根因坐实 + 双层修复成功(2026-06-25 10:40)
- **代码实锤**(用户要求):`verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:497-506`(原 500-502)
  ```python
  # 原公式(单轮假设,对 agent loop 多轮是 bug):
  max_tokens = min(response_length, prompt_length + response_length - len(prompt_ids))
  max_tokens = max(1, min(max_tokens, max_possible_tokens))  # clamp ≥1
  ```
  - prompt 涨到 prompt_length+response_length(24576)→ max_tokens=min(8192,24576-24576)=0 → clamp 1 → **1-token 退化**
  - 注释自己写 "prevent OOM in multi-turn rollouts",实际反而**导致**多轮退化
- **双层限制叠加**(解释为何之前全错):
  - 层1(max_tokens 公式):阈值固定 24576,**独立于 max_model_len** → mmlen40960 实验(只改 max_model_len)无效,55% 退化
  - 层2(max_model_len):修复公式后阈值=max_model_len;max_model_len=25600 时阈值 25600(只比 24576 多 1024)→ fix-only 实验 49% 退化
- **双层修复**(两个都改才有效):
  1. vllm_async_server.py:500 改 `max_tokens = min(response_length, max_possible_tokens)`(= max_model_len - prompt)
  2. parallel_infer.py:69 加 `MAX_MODEL_LEN` env override,设 40960(Qwen3-8B 原生 context)
- **验证结果(fix + MAX_MODEL_LEN=40960, tp2, 32并发, 16样本)**:
  - **completion=1 占比:0%**(545 个 completion 全部 >1,max 56,无 8192 截断)
  - 对比:c32 92% / tp2 87%traj / mmlen40960 55% / fix-only 49% → **彻底消除**
  - max_seq_len=40960,concurrency 4.76x,沙箱正常起
- **根因诊断链(5 次修正)**:KV池不足❌ → preempt死锁❌ → tp2增池❌ → max_model_len撞墙❌ → **max_tokens公式×max_model_len双层**✅。用户两次关键质疑命中:"输出长度配置不对"+"代码实锤"
- **下一步**:等 RM Score 确认实际成功率;若 32 并发稳,4 机 × tp2 × 32 = 总 128 可行(治本,非降并发)
- **Files 改动**(待 commit,当前计划相关):
  - verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:497-513(max_tokens 公式修复)
  - examples/agent_interaction/parallel_infer.py:69-75(MAX_MODEL_LEN env override)
