#!/bin/bash
# Install uni-agent dependencies. Can be executed from anywhere:

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

git -C "${PROJECT_ROOT}" submodule update --init --recursive
pip install --no-deps -e "${PROJECT_ROOT}/verl"
pip install swe-rex loguru pydantic pydantic_settings boto3
pip install --no-cache-dir swebench
