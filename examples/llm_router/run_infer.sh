#!/usr/bin/env bash
# KV-cache-aware router agent inference on the verl framework + TQ path.
#
# A thin wrapper around parallel_infer_verl_kvc.py: sets up the Ray-worker
# environment (AKernel sandbox creds + observability env), then forwards all
# CLI flags verbatim to parallel_infer_verl_kvc.py's argparse — see
# `bash run_infer.sh --help`.
#
#   bash examples/llm_router/run_infer.sh --model-path /path/to/model \
#       --data-path /path/to/swe_bench.parquet \
#       --task-config examples/llm_router/task_config_openyuanrong.yaml \
#       --max-samples 1 --kv-events
#   bash examples/llm_router/run_infer.sh --help   # full flag list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# The FQN-injected router (uni_agent.llm_router) resolves from the repo-root
# package; verl resolves from its own installed package. Prepend both.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Environment variables (exported for Ray-spawned workers) ──────────────
# These are read via os.environ inside Ray tasks, not CLI flags.
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-}"
export AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

# Forward every CLI flag to parallel_infer_verl_kvc.py (the single source of
# flags, defaults, types, and --help). No shell-side defaults — argparse owns them.
python examples/llm_router/parallel_infer_verl_kvc.py "$@"
