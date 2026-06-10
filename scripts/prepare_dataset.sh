#!/bin/bash
# Prepare SWE-Bench Verified dataset. Can be executed from anywhere:
#   bash scripts/prepare_dataset.sh
#   DEPLOYMENT=modal bash scripts/prepare_dataset.sh
#
# DEPLOYMENT determines the image name written into the parquet:
#   - modal:    swebench/sweb.eval.x86_64.<project>_<number>  (works with local docker, docker auto-pulls)
#   - vefaas:   Alibaba Cloud veFaaS image names               (only for veFaaS deployment)
#   - local:    not implemented yet
#
# For local docker sandbox, use DEPLOYMENT=modal (the default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

DEPLOYMENT=${1:-modal}
SAVE_DIR="${SCRIPT_DIR}"

echo "=== Prepare dataset ==="
echo "DEPLOYMENT : ${DEPLOYMENT}"
echo "Save dir   : ${SAVE_DIR}"

DEPLOYMENT=${DEPLOYMENT} python "${PROJECT_ROOT}/examples/data_preprocess/swe_bench_verified.py" \
    --local-save-dir "${SAVE_DIR}"

OUTPUT="${SAVE_DIR}/swe_bench_verified_${DEPLOYMENT}.parquet"
if [ ! -f "${OUTPUT}" ]; then
    echo "ERROR: Expected output not found: ${OUTPUT}" >&2
    exit 1
fi
echo "✅ Dataset ready: ${OUTPUT}"