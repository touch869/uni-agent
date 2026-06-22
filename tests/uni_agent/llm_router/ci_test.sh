#!/usr/bin/env bash
# ci_test.sh — full CI test suite for llm_router.
#
# Phase 1 (unit tests): config, strategies, balancer — no vLLM required.
# Phase 2 (integration): PollingCollector & KVEventCollector against a real
#              vLLM service (needs GPU + model download).
#
# Environment variables (phase 2 only):
#   VLLM_MODEL       — HuggingFace model ID or local path (default: Qwen/Qwen3-4B)
#   VLLM_HOST        — host to bind vLLM server (default: 127.0.0.1)
#   VLLM_PORT        — port for vLLM server (default: 8000)
#   ZMQ_SUB_PORT     — ZMQ PUB socket port (default: 5555)
#   ZMQ_REPLAY_PORT  — ZMQ ROUTER replay port (default: 5556)
#
# Usage:
#   ./ci_test.sh                                                                 # run all tests
#   ./ci_test.sh --unit                                                          # run only unit tests (no GPU)
#   CUDA_VISIBLE_DEVICES=6,7 ./ci_test.sh --polling                              # run only PollingCollector
#   ./ci_test.sh --kv-event                                                      # run only KVEventCollector
#   CUDA_VISIBLE_DEVICES=6,7 ./ci_test.sh --unit --polling                       # run unit + polling (no kv)
#   ./ci_test.sh --polling --kv-event                                            # run both integration, skip unit
#   VLLM_MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=6,7 ./ci_test.sh               # custom model

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
# NOTE: Defaults must match the hardcoded values in the source:
#   PollingCollector:       server_address = ["127.0.0.1:8000"]
#   ZMQEventCollector:      kv_event_address = {"127.0.0.1:8000": ["127.0.0.1:5555", "127.0.0.1:5556"]}
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export ZMQ_SUB_PORT="${ZMQ_SUB_PORT:-5555}"
export ZMQ_REPLAY_PORT="${ZMQ_REPLAY_PORT:-5556}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Select which tests to run ────────────────────────────────────────────
# Default: run all three groups.  Each flag is an independent toggle — you
# can combine them freely (e.g. --unit --polling = unit + polling, no kv).
# Bare ./ci_test.sh with no flags still runs everything.
RUN_UNIT=false
RUN_POLLING=false
RUN_KV_EVENT=false
RUN_PROVIDER=false
RUN_ALL=true          # true when no specific flags are given

for arg in "$@"; do
    case "$arg" in
        --unit)      RUN_ALL=false; RUN_UNIT=true ;;
        --polling)   RUN_ALL=false; RUN_POLLING=true ;;
        --kv-event)  RUN_ALL=false; RUN_KV_EVENT=true ;;
        --provider)  RUN_ALL=false; RUN_PROVIDER=true ;;
    esac
done

if $RUN_ALL; then RUN_UNIT=true; RUN_POLLING=true; RUN_KV_EVENT=true; RUN_PROVIDER=true; fi

echo "=== llm_router CI tests ==="
echo "  Model        : ${VLLM_MODEL}"
echo "  Host         : ${VLLM_HOST}"
echo "  Port         : ${VLLM_PORT}"
echo "  ZMQ Sub Port : ${ZMQ_SUB_PORT}"
echo "  ZMQ Replay   : ${ZMQ_REPLAY_PORT}"
echo ""

# ── Phase 1: Unit tests (no vLLM, no GPU) ────────────────────────────────
if $RUN_UNIT; then
    echo "--- Phase 1: Unit tests ---"
    python -m pytest \
        "${SCRIPT_DIR}/test_config.py" \
        "${SCRIPT_DIR}/test_strategies.py" \
        "${SCRIPT_DIR}/test_balancer.py" \
        "${SCRIPT_DIR}/test_balancer_integration_on_cpu.py" \
        -v
fi

# ── Phase 2: Integration tests (need real vLLM + GPU) ────────────────────
if $RUN_POLLING; then
    echo "--- Phase 2a: PollingCollector integration ---"
    VLLM_PORT=${VLLM_PORT} python -m pytest "${SCRIPT_DIR}/collectors/test_vllm_polling_collector.py" -v
fi

if $RUN_KV_EVENT; then
    echo "--- Phase 2b: KVEventCollector integration ---"
    VLLM_PORT=${VLLM_PORT} ZMQ_SUB_PORT=${ZMQ_SUB_PORT} ZMQ_REPLAY_PORT=${ZMQ_REPLAY_PORT} \
        python -m pytest "${SCRIPT_DIR}/collectors/test_vllm_kv_event_collector.py" -v
fi

if $RUN_PROVIDER; then
    echo "--- Phase 2c: RouteDataProvider.get_gpu_prefix_hit_rate integration ---"
    VLLM_PORT=${VLLM_PORT} ZMQ_SUB_PORT=${ZMQ_SUB_PORT} ZMQ_REPLAY_PORT=${ZMQ_REPLAY_PORT} \
        python -m pytest "${SCRIPT_DIR}/collectors/test_route_data_provider_gpu_prefix_hit_rate.py" -v
fi

echo "=== All tests done ==="
