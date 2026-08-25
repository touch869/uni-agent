# Agent Aware Router Examples

Experiment matrix drivers for the KV-cache-aware router: sweep the router over
(concurrency × context × load-threshold) against SWE-bench Verified, retrying each
run until it completes.

For setup and concepts, see the documentation:

- [Run the Agent Aware Router](https://uni-agent.readthedocs.io/en/latest/quickstart/agent-aware-router.html)
- [Agent Aware Router concepts](https://uni-agent.readthedocs.io/en/latest/concepts/agent-aware-router.html)

## Files

- `single-node.sh`: single-node matrix driver (the main entry).
- `multi-node.sh`: the same matrix on a multi-node Ray cluster.
- `run_infer.sh` / `run_infer.py`: one-shot inference driver the matrix calls per run (`--help` for the full flag set).
- `task_config_mini_swe_agent.yaml`: task/agent config used by the matrix (mini-swe-agent in an openyuanrong sandbox).

## Environment

Packages:

- [verl](https://github.com/verl-project/verl)
- [uni-agent (router branch)](https://github.com/touch869/uni-agent/tree/router)
- [rl-insight (verl agentic rollout dashboard branch)](https://github.com/touch869/rl-insight/tree/feat/verl-agentic-rollout-dashboard)

rl-insight needs a one-time install before the driver can start it:

```bash
rl-insight server install
```

> The driver starts/stops the rl-insight server around each run; set `VERL_RL_INSIGHT_ENABLE=0` to disable it.

OpenYuanrong sandbox (the only reverse-tunnel provider) — export these before running:

```bash
export DEPLOYMENT="openyuanrong"
export OPENYUANRONG_SERVER_ADDRESS="<server-address>"
export OPENYUANRONG_TOKEN="<token>"
export TUNNEL_SSL_VERIFY="0"
```

> For the platform itself, see the [OpenYuanRong docs](https://docs.openyuanrong.org/zh-cn/latest/index.html). The sandbox is backed by [AKernel](https://github.com/inclusionAI/AKernel/blob/main/AGENTS.md) (`akernel_sdk`), which is where `<server-address>` / `<token>` come from.

## Single-node usage

Prerequisites: a host with >= 8 GPUs (`--n-gpus-per-node` is pinned at 8), a local
model path, and a preprocessed dataset:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir /path/to/swe_agent
```

```bash
DEVICE=gpu TP=2 MODEL=/path/to/model DATASET=/path/to/swe_agent/swe_bench_verified.parquet \
CONCURRENCYS="16" CONTEXTS="16384" LTS="0.7" MAX_SAMPLES=4 N=2 bash examples/agent_aware_router/single-node.sh
```

That smoke run sweeps one matrix cell; the defaults sweep
`CONCURRENCYS="16 24 32 128"` × `CONTEXTS="16384 32768 64000 128000"` × `LTS="0.7 0.9"`.
Other knobs (defaults): `DEVICE=ascend` (`gpu` for NVIDIA), `TP=4`, `MAX_SAMPLES=64`,
`RES_LEN=8000`, `N=8`.

Each run writes `infer-${DEVICE}-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}.log`
and is retried until the `inference summary` sentinel lands in its log; before every
attempt the script kills leftover `run_infer`/ray processes and clears the device.
A sticky baseline arm is planned to complete the sticky-vs-kvcaware comparison.
