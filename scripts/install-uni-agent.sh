#!/bin/bash
# Install uni-agent dependencies. Can be executed from anywhere:

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

git -C "${PROJECT_ROOT}" submodule update --init --recursive
pip install --no-deps -e "${PROJECT_ROOT}/verl" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e "${PROJECT_ROOT}" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install swe-rex loguru pydantic pydantic_settings boto3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir swebench -i https://pypi.tuna.tsinghua.edu.cn/simple
