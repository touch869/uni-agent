#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-hgq-swe}"
IMAGE_NAME="${IMAGE_NAME:-verlai/verl:vllm018.dev1}"
SHM_SIZE="${SHM_SIZE:-10g}"

# 停止旧容器
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --shm-size=${SHM_SIZE} \
  -v /data1:/data1 \
  -v /tmp:/tmp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  --entrypoint sleep \
  ${IMAGE_NAME} \
  infinity

# 验证
echo "=== 验证 ==="
docker exec ${CONTAINER_NAME} nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "✅ 容器 ${CONTAINER_NAME} 就绪 (GPU device=${GPU_DEVICE})"


