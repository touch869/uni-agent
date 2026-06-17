#!/usr/bin/env bash
# ci_test.sh — run VLLMPollingCollector & VLLMKVEventCollector integration tests
#              against a real vLLM service.
#
# Environment variables:
#   VLLM_MODEL       — HuggingFace model ID (default: Qwen/Qwen3-4B)
#   VLLM_HOST        — host to bind vLLM server (default: 127.0.0.1)
#   VLLM_PORT        — port for vLLM server (default: 8000)
#   ZMQ_SUB_PORT     — ZMQ PUB socket port (default: 5555)
#   ZMQ_REPLAY_PORT  — ZMQ ROUTER replay port (default: 5556)
#
# Usage:
#   ./ci_test.sh                                        # run all tests with defaults
#   ./ci_test.sh --polling                              # run only polling collector tests
#   ./ci_test.sh --kv-event                             # run only KV event collector tests
#   VLLM_MODEL=Qwen/Qwen3-8B ./ci_test.sh               # run with custom model

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

echo "=== llm_router collector CI tests ==="
echo "  Model        : ${VLLM_MODEL}"
echo "  Host         : ${VLLM_HOST}"
echo "  Port         : ${VLLM_PORT}"
echo "  ZMQ Sub Port : ${ZMQ_SUB_PORT}"
echo "  ZMQ Replay   : ${ZMQ_REPLAY_PORT}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Select which tests to run ────────────────────────────────────────────
RUN_POLLING=true
RUN_KV_EVENT=true

if [[ "${1:-}" == "--polling" ]]; then
    RUN_KV_EVENT=false
elif [[ "${1:-}" == "--kv-event" ]]; then
    RUN_POLLING=false
fi

if $RUN_POLLING; then
    echo "--- Running PollingCollector tests ---"
    python -m pytest "${SCRIPT_DIR}/collectors/test_vllm_polling_collector.py" -v
fi

if $RUN_KV_EVENT; then
    echo "--- Running KVEventCollector tests ---"
    python -m pytest "${SCRIPT_DIR}/collectors/test_vllm_kv_event_collector.py" -v
fi

echo "=== All tests done ==="
