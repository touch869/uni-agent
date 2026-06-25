# Progress Log

## 2026-06-25

### 已完成
- 通读 uni-agent / infer_multi.sh 完整流程（脚本 → parallel_infer → verl AgentLoopManager/Worker → UniAgentLoop → AgentInteraction step 循环 → LLMServerClient 路由到 vLLM 副本）。
- 盘点沙箱调用面：AgentEnv 是唯一门面，10 个协议方法（见 findings.md §1）。
- brainstorming 完成，设计获用户认可：
  - 方案 A（挂最底层 AbstractDeployment/AbstractRuntime，`type: mock` 判别联合接入）
  - 手写代表性模板池（真实样例结构 + 自定权重）
  - 默认随机 + seed 可复现 + observation_scale
  - 三种失败模拟（observation 内容失败 / 超时 / 终端死亡）
- 采样真实轨迹（8.92.9.147:/data1/hgq/agentic_log）校准模板内容结构。
- 创建 planning/ 三文件（task_plan / findings / progress）。

### 决策
- 见 task_plan.md「决策记录」D1–D5。

### 下一步
- 进入实现：Phase 0（基建/配置骨架）→ Phase 1（路由/渲染）→ …
- 每个 Phase 完成后回填 task_plan 进度表。

### 执行环境（重要）
- **单测基地 = 150**：`hgq-swe-mock` 容器（GPU 4,5），repo `/data1/hgq/uni-agent-mock`，分支 `mock-sandbox`，基线 `e69f13a`。
- **压测基地 = 147**：`hgq-swe-mock` 容器（全 8 卡），repo `/data1/hgq/uni-agent-mock`，分支 `mock-sandbox`，基线 `b7cb982`（含 CONCURRENCY env override，planning2 16/32 并发压测要用）。
  - 两台都是 `cp -a` 独立副本（各自独立 `.git`），与共享盘 main 隔离。
  - 147 基线 b7cb982 = 150 e69f13a 的"姊妹分叉"（都在 mooncake 方向，147 多了 CONCURRENCY override）。`deployment/config.py`+`__init__.py` 两台一致，mock 改动零冲突直接应用。
  - **同步策略 B**：本地 Write → tar/scp 到 .147/.150:/tmp → docker cp 进容器 → pytest → git commit。
- 文件传递：本地 Write → scp 到 :/tmp → docker cp 进容器（避免多层 ssh+docker heredoc 引号问题）。
- 现有 host_runtime 基线测试有 3 个 fail（timeout/interrupt，容器环境特性，与本 mock 无关），不阻塞。
- ⚠️ ssh→docker `bash -lc "..."` 双引号内 `(` `)` 会被解析为子shell，echo 文案别带括号；commit message 含 `<...>` 用 `-F file` 传递。
- **51 个 mock 测试两台全绿**（150 + 147）。

### TDD 进度
- [x] **Phase 1** RED→GREEN：`MockRuntime._route` + `run_in_session` 渲染。
      - 路由：23 个 parametrize case（`_ROUTE_RULES` 顺序敏感）。
      - 渲染：模板池按 weight 抽样；`seed` 可复现（同 seed 字节级一致）/ `None` 默认随机；`observation_scale` 缩放。
      - finish/install 固定输出（让 AgentEnv.install_tools 通过）。
      测试：test_mock_routing.py（23）+ test_mock_render.py（7）。
- [x] **Phase 2** RED→GREEN：MockDeployment + 判别联合注册 + AgentEnv 集成。
      - `mock/config.py`：MockDeploymentConfig（seed/scale + timeout/terminal_dead 配置位，逻辑待 Phase 4）。
      - `mock/deployment.py`：MockDeployment（start/stop/is_alive/runtime，runtime 未 start 时抛 DeploymentNotStartedError，与 host 一致）。
      - `config.py` 联合 + `__init__.py` 懒导出注册 `type: mock`。
      - 集成：真实 `AgentEnv` 在 mock 部署下跑通 `start → install(which/chmod/export) → run_action(editor:view/finish) → close`，零改上层。
      测试：test_mock_config.py（5）+ test_mock_agent_env_integration.py（2）。**共 37 passed。**
- [ ] 下个循环：Phase 3（模板池内容丰富化——真实样例结构 + 自定权重 + 长尾）或 Phase 4（失败模拟：observation 失败 / 超时 / 终端死亡）。
- [x] **Phase 4** RED→GREEN：失败模拟三件套（8 测试，共 45 mock 测试过）。
      - **超时**：`run_in_session` sleep(delay) → 抛 swerex `CommandTimeoutError` → env.py 转 `ActionTimeoutError`（可恢复，扣 timeout_budget）。seed 可复现。
      - **终端死亡**：设 `_dead` 标志。命中时该命令抛 `CommandTimeoutError`（触发 env.py interrupt+探活）→ `BashInterruptAction` 也抛异常（让 env.py 走探活分支）→ 探活 `run_in_session` 返回不含 marker 的空内容 → **env.py 自然 raise `TerminalNotAliveError`**（致命，episode 终止）。这是唯一能产生正确 `terminal_dead` 语义的路径（TerminalNotAliveError 由 env.py 探活逻辑抛出，runtime 不直接抛）。
      - **observation 层失败**：模板池 traceback/cmd-not-found（exit_code≠0），沙箱不抛异常。
      - 超时/终端死亡用**独立 RNG 流**（`__timeout__`/`__terminal_dead__`），install/finish 豁免。
      测试：test_mock_failure.py（8）。

### Phase 5b 对照压测结果（147, Qwen3-8B, 4 prompt × n=4 = 16 traj, TP=2, MAX_MODEL_LEN=40960）

| 指标 | 基线 local docker | mock 沙箱 |
|---|---|---|
| 耗时 | ~12 min | ~18 min |
| generation throughput | 168-232 tok/s | 256-268 tok/s（峰值更高，沙箱零延迟不阻塞 LLM） |
| finished 正常 submit | 有 | **7/16** ✅ |
| max_step_limit（跑满100步） | — | 8/16 |
| terminal_dead/崩溃 | 0 | **0** ✅ |
| prefix cache hit | 62-85% | 99.4%（mock obs 短且重复，命中超高） |
| Mean RM Score | 0.0625（真实） | 0.0000（预期：假沙箱不做真题） |

**结论**：mock 沙箱完整复现真实 agent loop（7 条 finished、0 崩溃），LLM 服务层表现健康。**mock 作为 LLM 服务层压测工具成功**——该复现的（轨迹结构、LLM 负载、无沙箱崩）都对；不该复现的（真实通过率）自然为 0。

**修复的 2 个压测中暴露的 bug**：
1. **docker overlay 损坏**（`[[docker-overlay-stale-mount-enospc]]`）：147 daemon overlay2 状态损坏 → 基线首轮 7/16 沙箱起不来 → RM=0 误判。`systemctl restart docker` 修复。
2. **MockDeploymentConfig extra=forbid**（commit 5ef1295）：数据集 tools_kwargs 带 docker-only 字段(image/command)→ mock 校验失败。改 extra=ignore + 回归测试。

**轨迹分析坑**（重要）：grep trajectory 的 exit_reason 会抓到每步的所有 reason（含中间步的 completed/terminal_dead 字样），误判成"卡死"。**正确做法**：只看 trajectory[-1].exit_reason（最后一步）。实际 mock 16 traj：7 finished / 8 max_step_limit / 1 token_limit / **0 terminal_dead**。
- `TerminalNotAliveError` 定义在 `uni_agent/interaction/env.py`，**由 env.py 在"超时+探活失败"时自己抛**，不是 runtime 抛。mock 必须经 env.py 探活路径才能产生正确 terminal_dead 语义：runtime 抛 CommandTimeoutError + interrupt 失败 + 探活返回无 marker。
- 后果链差异：**超时=可恢复**（扣 budget，done=False，继续）；**终端死亡=致命**（step exit_reason=terminal_dead，done=True，episode 当场终）。

### 提交记录（mock-sandbox 分支）
- `[mock] Phase 1: MockRuntime route + render with reproducible seeding`
- `[docs] add mock-sandbox planning files`
- `[mock] Phase 2: MockDeploymentConfig + discriminated-union registration + AgentEnv integration`
- `[docs] update progress: Phase 1+2 done`
- `[mock] Phase 4: failure simulation (timeout + terminal death + obs-layer failures)`

### 待确认（实现前）
- 模板池载体：内置 `templates.py` dict vs 独立 `mock_observations.yaml`（倾向独立 yaml，方便单独调比例）。
- read_file/write_file：临时目录真读写 vs 空响应（倾向临时目录，零依赖但更真）。
