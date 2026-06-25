# Findings: Mock 沙箱部署 研究/发现

## 1. 沙箱门面 = AgentEnv（唯一入口）

`uni_agent/interaction/env.py::AgentEnv` 是所有沙箱调用的唯一门面。
所有调用走 `self.deployment`（`AbstractDeployment`）或 `self.deployment.runtime`（`AbstractRuntime`）。

### Deployment 层调用点
| 位置 | 调用 | 作用 |
|---|---|---|
| `start()` | `deployment.start(max_retries)` | 起容器/swerex |
| `start()` 内 | `communicate(post_setup_cmd)` | post-setup |
| `install_tools` | `communicate("export PATH=…")` | 设 PATH |
| `install_tools` | `copy_to_container(src,tgt)` | 拷工具脚本 |
| `install_tools` | `communicate("chmod +x …")` | 加可执行 |
| `install_tools` | `communicate(install_cmd)` | pip 装 |
| `install_tools` | `communicate("which {name}")` | 存在性校验 |
| `close()` | `deployment.stop()` | 关容器 |

### Runtime 层调用点（经 communicate/copy_to_container/interrupt_session 间接）
| 方法 | 实际调用 | 用在哪 |
|---|---|---|
| `runtime.run_in_session(BashAction)` | **核心** | run_action / 所有 communicate |
| `runtime.run_in_session(BashInterruptAction)` | 中断超时命令 | interrupt_session |
| `runtime.execute(Command)` | 一次性命令（mkdir） | copy_to_container |
| `runtime.upload(UploadRequest)` | 传文件 | copy_to_container |
| `runtime.read_file(ReadFileRequest)` | 读文件 | read_file |
| `runtime.write_file(WriteFileRequest)` | 写文件 | write_file |

→ MockRuntime 必须实现上述 6 + `is_alive`/`create_session`/`close_session`/`close`，共 10 个方法（多数桩）。

## 2. 部署类型判别联合（接入点）

`uni_agent/deployment/config.py`：`DeployConfig = Annotated[... | ... , Field(discriminator="type")]`。
每种部署一个 `XxxConfig`（`type: Literal["xxx"]`）+ `get_deployment(run_id)` 工厂 + `__init__.py` 懒导出。
现有：host / local_native / local / local_attach / modal / vefaas / remote。
→ 加 `MockDeploymentConfig`（`type: Literal["mock"]`）完全同构接入。

## 3. 工具 → bash 命令的映射（路由键来源）

`uni_agent/interaction/tools_manager.py::get_tool_bash_command`：
- `submit` → `echo '<<<Finished>>>'`（固定）
- `execute_bash` → `func_params["command"]` 原样（任意 bash）
- `str_replace_editor` → `str_replace_editor --command <X> --path ... --file_text ...`（CLI argv，可解析）
- `lark-cli` → `lark-cli <command>`
- 其它 → `<name> --<key> <val> ...`

→ mock 收到 bash 字符串后可确定性地路由到工具类型（submit/editor 干净；execute_bash 需按首词子路由）。

## 4. 真实 observation 样例（从 8.92.9.147:/data1/hgq/agentic_log 采样，模板内容参考）

抓取自 1500 条 SWE-bench 轨迹的 `interaction_result.json`，结构确认：

- **pytest 输出**：超长环境 banner（platform/python version/package versions/plugins）+ collecting + 结果行 + summary
- **python traceback**：`Traceback (most recent call last):` + 逐帧 `File "...", line N, in <fn>` + 末行 `ExceptionType: msg`
- **find 输出**：纯文件路径列表（`/testbed/.../foo.py` 每行一个）
- **editor view**：`"Observation:\n"` 前缀 + `cat -n` 行号格式
- **grep 无匹配**：`"Your command ran successfully and did not produce any output."`（env.py 兜底）
- **所有 observation**：经 env.py `run_action` 包成 `"Observation:\n<content>"`，长内容裁剪到 100k + `<response clipped>`

样例原文见 brainstorming 对话，已固化进 Phase 3 模板池要求。

## 5. 关键发现：日志频率不可信，内容可信

- **tool status 全是 `ok`**（1500 条样本）：trajectory 里**没有** status=timeout/terminal_dead 的样本。
- **observation 内容失败高频**：python_traceback(1690) / misc_error(754) / no_such_file(442) / connection_err(278)。
- **run.log 有 929 个 CRITICAL**：是 KV 显存不足 bug 导致 agent_loop 失败，连带产生大量 `format_error` exit。
- 用户明确：**`format_error`/`completed` 的高频比例是 KV bug 副作用 → 模板权重不照搬日志，自定合理值（成功:失败≈8:2）。但真实 observation 文本结构可用作模板内容。**
- **超时/终端死亡这批日志没在 trajectory 出现**（被 KV bug 掩盖），但 env.py 有这两条真实 except 分支 → perf 要压到慢路径，故按低概率模拟。

## 6. observation 长度量级（仅作模板尺寸参考，权重另定）

600 条采样（注意：频率受 KV bug 污染，仅看量级）：
- `str_replace_editor`：p50=553 / p90=4864 / p99=16602（**预填充 KV 压力主力**）
- `execute_bash`/python script：p50=190 / p90=1275 / p99=1600（高频小输出）
- `pytest`：p50=459 / p90=1751（没预想的那么大）
- 长尾：`ls`/`grep`/`export` 偶发 100k+（误操作 dump，需覆盖）

→ editor 池要覆盖 ~300 到 ~16000 多档；python 池以小输出为主；单独留 100k 巨无霸长尾模板。

## 7. env.py 异常分支（失败模拟要覆盖）

`run_action` 的 except：
- `CommandTimeoutError` → interrupt_session → 探活 → 抛 `ActionTimeoutError`（observation 是超时提示文本）→ timeout_budget--
- `BashIncorrectSyntaxError` → `ActionIncorrectSyntaxError`
- （探活失败）→ `TerminalNotAliveError` → terminal_dead，后续 tool call 全 skipped

→ mock 要能真实抛 `CommandTimeoutError`（swerex 异常）和 `TerminalNotAliveError`，让 env.py 走完整慢路径。
