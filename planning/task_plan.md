# Task Plan: Mock 沙箱部署（MockDeployment / MockRuntime）

> 目标：新增 `type: mock` 部署，把整个沙箱（docker/swe-rex）调用 mock 掉，
> 供 **性能压测** 使用。LLM（vLLM 副本 + router）真跑，只 mock 沙箱。
> 设计已通过 brainstorming 并获用户认可（见本目录 findings.md）。

## 设计决策（已锁定）

- **挂载点**：方案 A —— 实现 swerex `AbstractDeployment`/`AbstractRuntime`，
  新增 `MockDeploymentConfig` 注册进 `DeployConfig` 判别联合（`type: mock`）。
  `AgentEnv` 之上零改动，整条生产代码路径原封不动地跑，只换最末端的 bash 执行。
- **observation 来源**：手写代表性模板池，内容结构搬真实 SWE-bench 样例，
  **权重自定合理值**（不照搬日志频率——日志高频 `format_error` 是 KV 显存 bug 副作用）。
- **随机性**：默认随机抽样；`seed` 设整数 → 可复现。随机源按 `(路由键, 第k次)` 独立取种。
- **`observation_scale`**：全局长度缩放，默认 1.0（敏感度扫描用）。
- **失败模拟**（三种全加，概率可配、seed 可锁）：
  1. observation 层失败（内容：traceback / no such file / cmd not found，exit_code≠0，不抛沙箱异常）
  2. 命令超时（`run_in_session` 主动 sleep + 抛真实 `CommandTimeoutError` → 走 env.py 超时分支）
  3. 终端死亡（抛真实 `TerminalNotAliveError` → terminal_dead exit）

## 路由规则（MockRuntime 收到的 bash 字符串 → 路由键）

| 命令特征 | 路由键 | 来源 |
|---|---|---|
| `echo '<<<Finished>>>'` | `finish` | submit/finish 工具固定输出 |
| `str_replace_editor --command view ...` | `editor:view` | 解析 argv |
| `str_replace_editor --command create/str_replace/insert/undo_edit` | `editor:edit` | 解析 argv |
| `which <name>` / `chmod` / `export PATH` / `mkdir` / `pip install` | `install` | install_tools 阶段，空 success |
| `python*.py pytest|pytest ...` | `test_output` | |
| `python *.py` / `python3 ...` | `python_script` | |
| `find` / `ls` | `listing` | |
| `grep` | `search` | |
| `cat` / `head` / `tail` | `file_view` | |
| 其它 | `default` | |

## 协议方法（全部从 env.py 调用面反推，无遗漏）

```
MockRuntime:
  run_in_session(action)   # BashAction -> route+render ; BashInterruptAction -> Observation(exit=130)
  execute(Command)         # CommandResponse(空, exit=0)  [install mkdir]
  upload(UploadRequest)    # UploadResponse()  [tool/skill 拷贝 no-op]
  read_file / write_file   # 临时目录真读写 / 空响应
  is_alive / create_session / close_session / close   # 状态桩
MockDeployment:
  from_config / start / stop / is_alive / add_hook / runtime(property)
```

---

## Phase 0 — 基建与配置骨架  `[ ]`
- [ ] 新建 `uni_agent/deployment/mock/` 包（`__init__.py`）
- [ ] `mock/config.py`：`MockDeploymentConfig`（pydantic，`type: Literal["mock"]`，字段：
      `seed: int|None=None`, `observation_scale: float=1.0`, `templates: str="builtin"`,
      `timeout: TimeoutCfg`, `terminal_dead: ProbCfg`），`model_config=ConfigDict(extra="forbid")`
- [ ] 在 `uni_agent/deployment/config.py` 的 `DeployConfig` 联合类型 + `__init__.py` 懒导出里加入 `MockDeploymentConfig`

## Phase 1 — MockRuntime 核心路由 + 渲染  `[ ]`
- [ ] `mock/deployment.py`：`MockRuntime`（继承 `AbstractRuntime`）
- [ ] `_route(command) -> str`：解析 bash 字符串 → 路由键（argv 解析 + 正则）
- [ ] 模板池加载：`builtin` 内置（Phase 3），支持外部 yaml override
- [ ] `_render(key, command) -> Observation`：按 `(seed, key, 第k次)` 抽模板 → 乘 scale → 包成 `Observation`
- [ ] 随机源：每 `(路由键)` 一个 `random.Random(seed^hash(key))` 计数器序列
- [ ] 协议桩方法：`execute/upload/read_file/write_file/is_alive/create_session/close_session/close`

## Phase 2 — MockDeployment + 接入  `[ ]`
- [ ] `MockDeployment.from_config/start/stop/is_alive/add_hook`，`runtime` property 返回 MockRuntime
- [ ] `start()` 不拉进程、不连 docker、不装 swe-rex（构造 MockRuntime + 标记 started）
- [ ] 验证：`AgentEnv(type: mock)` 能跑通 `start → install_tools → install_skills → run_action → close`

## Phase 3 — 模板池数据（真实样例结构 + 自定权重）  `[ ]`
- [ ] `mock/templates.py`（或 `mock_observations.yaml`）：每个路由键多条模板 + weight
- [ ] 内容搬真实样例：pytest banner/FAILED、python traceback 栈帧、find 文件列表、
      editor view `cat -n` 行号格式、"Observation:\n" 前缀
- [ ] 权重自定（成功:失败 ≈ 8:2，长尾 100k 巨无霸保留低权重覆盖误操作）
- [ ] editor 池覆盖小(~300)/中(~600)/大(~4800)/超大(~16000)
- [ ] python_script 池：成功小输出(主) + traceback + command not found

## Phase 4 — 失败/异常模拟  `[ ]`
- [ ] observation 层失败：已在 Phase 3 模板池里（exit_code≠0 的模板）
- [ ] 命令超时：`run_in_session` 命中 `timeout.probability` → `sleep(delay)` → 抛 `CommandTimeoutError`
- [ ] 终端死亡：命中 `terminal_dead.probability` → 抛 `TerminalNotAliveError`
- [ ] 概率事件吃 seed（可复现）；默认概率保守（timeout 0.05 / terminal_dead 0.01）

## Phase 5 — 示例配置 + 单测  `[ ]`
- [ ] `examples/agent_interaction/agent_config_mock.yaml`
- [ ] `scripts/infer_multi_mock.sh`（或文档说明：`AGENT_CONFIG=...agent_config_mock.yaml bash scripts/infer_multi.sh`）
- [ ] `tests/uni_agent/deployment/test_mock_deployment.py`：
      - 路由键判定（各类命令）
      - 默认随机 vs seed 可复现（同 seed 两次跑抽到同样序列）
      - observation_scale 生效
      - install_tools/which/chmod 返回 success 不挂
      - 超时/终端死亡按概率触发并走对 env.py 分支
      - read_file/write_file 临时目录往返

---

## 进度

| Phase | 状态 | 备注 |
|---|---|---|
| 0 基建/配置 | ⬜ 未开始 | |
| 1 路由/渲染 | ⬜ 未开始 | |
| 2 Deployment 接入 | ⬜ 未开始 | |
| 3 模板池 | ⬜ 未开始 | |
| 4 失败模拟 | ⬜ 未开始 | |
| 5 配置/测试 | ⬜ 未开始 | |

## 决策记录

- D1（06-25）：mock 挂最底层 AbstractDeployment/AbstractRuntime，不走 AgentEnv 短路 —— 保生产路径完整。
- D2（06-25）：模板权重不参考日志频率（KV bug 污染），但内容结构参考日志真实样例。
- D3（06-25）：默认随机 + seed 可锁 —— perf 基线随机、A/B 对比锁 seed。
- D4（06-25）：失败模拟三种全加（observation 内容失败 / 超时 / 终端死亡），覆盖 env.py 全部 except 分支。
- D5（06-25）：observation_scale 保留，默认 1.0。
