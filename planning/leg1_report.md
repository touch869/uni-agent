# 腿一报告：llm-router vs sticky 调度（8B SWE-Bench）

> 阶段四腿一（A vs C）结论。详细数据见 `findings.md`。

## 结论

**router（KVCAware, alpha=0.7）显著优于 verl 内置 sticky（global_sticky_inflight）**：
- 目标①（吞吐）：router ~1.9x 更快（23 vs 44 min / 16 samples，高压）
- 目标②（GPU prefix hit）：router 4.4x（高压）/ 1.7x（低压）

两种 KV 压力下 router 相对优势都稳定成立。

## 数据（8B TP=1 DP=8 max_num_seqs=16，A/C 严格可比）

| 场景 | 组 | 稳态 GPU hit | wall-time | RM |
|---|---|---|---|---|
| 高压 eviction（KV~99.5%，max_model_len=37888）| A sticky | 8.3% | 44.5 min | 0.0625 |
| | C router | 36.5% | 23.2 min | 0.0000 |
| 低压（KV~85%，历史）| A sticky | 49.3% | — | (被kill) |
| | C router | 83.0% | — | 0.0625 |

## 方法论

- **A baseline = verl `global_sticky_inflight`**（sticky-session + least-inflight，router.py:239），即用户要的"一般 sticky 调度"对照——**不是 round-robin**。腿一无需独立实现 sticky。
- **C router = KVCAwareBalancer**，`strategies/kvc_aware_strategy.yaml`：alpha=0.7（S_cache 主导）→ 主动把请求导向"已缓存该 prefix"的副本。
- 指标口径：稳态 GPU prefix hit（排冷启动后 2/3）+ wall-time（samples/时长）。StatLogger 的 `Avg generation throughput` mean 不可靠（含空闲）。

## 限制 / 待查

- RM Score 小样本（16）噪声，不作为判定。
- hit 绝对值依赖 KV 压力（低压高、高压低），但 router 相对优势稳定。
- A 的 gen_tput median=0.1（大量低效行）vs C 6.3——疑 sticky 下某些副本过载或 swe-rex 沙箱 hang 拖慢，待深查。
- **目标③（Mooncake 降重算）属腿二 D 组**，本报告不含。

## 关键纠正（相对早期认知）

- ❌ 早期"router 无 prefix 收益" → ✅ router vs sticky **有显著 prefix 收益**。错误来自把 router+mooncake（mooncake 0.18 没生效）数据当成 router 本身。
