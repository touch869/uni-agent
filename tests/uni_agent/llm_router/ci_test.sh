#!/usr/bin/env bash
# ci_test.sh — run VLLMPollingCollector integration tests against a real vLLM service.
#
# Environment variables:
#   VLLM_MODEL   — HuggingFace model ID (default: Qwen/Qwen3-4B)
#   VLLM_HOST    — host to bind vLLM server (default: 127.0.0.1)
#   VLLM_PORT    — port for vLLM server (default: 8100)
#
# Usage:
#   ./ci_test.sh                          # run with defaults
#   VLLM_MODEL=Qwen/Qwen3-8B ./ci_test.sh # run with custom model

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8100}"

echo "=== VLLMPollingCollector CI Test ==="
echo "  Model : ${VLLM_MODEL}"
echo "  Host  : ${VLLM_HOST}"
echo "  Port  : ${VLLM_PORT}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pytest "${SCRIPT_DIR}/collectors/test_vllm_polling_collector.py" -v
