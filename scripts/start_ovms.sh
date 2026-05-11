#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(pwd)/models"
CONFIG_DIR="$(pwd)/config"
IMAGE="openvino/model_server:2026.1-gpu"

if [ ! -d "$MODELS_DIR" ]; then
  echo "models/ not found. Run scripts/export_model.sh first." >&2
  exit 1
fi

echo "=== Pulling OVMS image: $IMAGE ==="
docker pull "$IMAGE"

echo "=== Starting OpenVINO Model Server 2026.1 (GPU) ==="

docker run -d --rm \
  -p 8000:8000 \
  --device /dev/dri \
  --group-add $(stat -c "%g" /dev/dri/render* | head -n 1) \
  -v "$MODELS_DIR":/workspace/models:ro \
  -v "$CONFIG_DIR":/workspace/config:ro \
  "$IMAGE" \
  --rest_port 8000 \
  --config_path /workspace/config/config.json \
  --enable_prefix_caching true \
  --rest_workers 1

echo "✅ OVMS 2026.1 started. Endpoint: http://localhost:8000/v3/chat/completions"
