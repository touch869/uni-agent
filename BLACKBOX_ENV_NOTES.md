# Blackbox Mini-SWE Environment Notes

Configured container:

- Container: `uni-agent-blackbox`
- Image: `ubuntu-lll:uni-agent-blackbox`
- Repo: `/workspace/uni-agent`
- Model smoke-test path: `/workspace/models/Qwen3-1.7B`
- Dataset: `/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet`
- Local wheel: `/workspace/wheels/akernel_sdk-0.9.13-py3-none-any.whl`

Run a minimal smoke test after setting AKernel credentials:

```bash
docker exec -it uni-agent-blackbox bash -lc '
source /etc/profile.d/uni-agent.sh
cd /workspace/uni-agent
AKERNEL_SERVER_ADDRESS="<server>" \
AKERNEL_TOKEN="<token>" \
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/workspace/models/Qwen3-1.7B \
DATA_PATH=/workspace/data/swe_agent/swe_bench_verified_openyuanrong.parquet \
MAX_SAMPLES=1 \
PROMPT_LENGTH=4096 \
RESPONSE_LENGTH=512 \
TP=1 \
N_GPUS_PER_NODE=1 \
GATEWAY_COUNT=1 \
MAX_CONCURRENT_SESSIONS=1 \
AGENT_MAX_TURNS=1 \
ROLLOUT_GPU_MEM_UTIL=0.45 \
bash examples/blackbox_recipes/run_mini_swe.sh
'
```

The upstream branch entry point is `examples/blackbox_recipes/mini_swe_agent/run_infer.sh`.
`examples/blackbox_recipes/run_mini_swe.sh` is a compatibility symlink.

Credential aliases:

- `OPENYUANRONG_SERVER_ADDRESS` is accepted as an alias for `AKERNEL_SERVER_ADDRESS`.
- `OPENYUANRONG_TOKEN` is accepted as an alias for `AKERNEL_TOKEN`.
- `akernel_sdk 0.9.13` was installed from the local wheel above.
- For HTTPS/443 YR gateways, `akernel_sdk 0.9.13` has been patched locally to use `wss://` for the tunnel WebSocket.
- `swebench 4.1.0` and `swe-rex 1.4.0` are installed for SWE-bench reward evaluation.

Quick environment check without launching vLLM:

```bash
docker exec -it uni-agent-blackbox bash -lc '
source /etc/profile.d/uni-agent.sh
cd /workspace/uni-agent
bash examples/blackbox_recipes/check_mini_swe_env.sh
'
```

This check should only report missing AKernel/OpenYuanRong credentials before the final sandbox smoke test.

Recommended smoke-test wrapper:

```bash
cd /workspace/uni-agent
cp examples/blackbox_recipes/openyuanrong.env.example examples/blackbox_recipes/openyuanrong.env
# Fill OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN in openyuanrong.env.
CUDA_VISIBLE_DEVICES=6 bash examples/blackbox_recipes/run_mini_swe_smoke.sh
```

The wrapper defaults to one GPU, one sample, Qwen3-1.7B, and the generated SWE-bench Verified OpenYuanRong parquet. Override any variable in the shell or in `openyuanrong.env`.


Latest smoke result:

- Command: `CUDA_VISIBLE_DEVICES=6 bash examples/blackbox_recipes/run_mini_swe_smoke.sh`
- Result: exit code `0`.
- Verified path: env check, vLLM load, Ray gateway, OpenYuanRong sandbox creation, agent run, reward worker import/execution path, sandbox cleanup.
- The one-turn smoke sample ended with agent `LimitsExceeded` and zero submission, which is a model/task outcome for the intentionally tiny smoke settings, not an environment setup failure.
- `ak list` after the run reported no running instances.
