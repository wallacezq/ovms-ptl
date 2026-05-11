#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-8B"
OUT_DIR="$(pwd)/models"
TOOLS_DIR="$(pwd)/tools"

# Long-context / 64K starting-point knobs (override via env)
WEIGHT_FORMAT="${WEIGHT_FORMAT:-int4}"
KV_CACHE_PRECISION="${KV_CACHE_PRECISION:-u8}"
# Starting point for 64K tests; increase/decrease based on your memory + concurrency
CACHE_SIZE="${CACHE_SIZE:-16}"
TARGET_DEVICE="${TARGET_DEVICE:-GPU}"

mkdir -p "$OUT_DIR" "$TOOLS_DIR"

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install uv first (see README)."; exit 1; }

echo "=== [1/4] Fetch OVMS 2026.1 export tools ==="
EXPORT_PY_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2026/1/demos/common/export_models/export_model.py"
REQ_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2026/1/demos/common/export_models/requirements.txt"

curl -L "$EXPORT_PY_URL" -o "$TOOLS_DIR/export_model.py"
curl -L "$REQ_URL" -o "$TOOLS_DIR/export_requirements.txt"

if [ ! -d .venv-export ]; then
  uv venv .venv-export --python 3.11
fi

echo "=== [2/4] Install export deps into .venv-export ==="
uv pip install --python .venv-export/bin/python -U pip setuptools wheel
uv pip install --python .venv-export/bin/python -r "$TOOLS_DIR/export_requirements.txt"

echo "=== [3/4] Export model to OpenVINO IR ==="
echo "Model: $MODEL_NAME"
echo "WEIGHT_FORMAT=$WEIGHT_FORMAT KV_CACHE_PRECISION=$KV_CACHE_PRECISION CACHE_SIZE=$CACHE_SIZE TARGET_DEVICE=$TARGET_DEVICE"

uv run --python .venv-export/bin/python "$TOOLS_DIR/export_model.py" text_generation \
  --source_model "$MODEL_NAME" \
  --weight-format "$WEIGHT_FORMAT" \
  --kv_cache_precision "$KV_CACHE_PRECISION" \
  --target_device "$TARGET_DEVICE" \
  --cache_size "$CACHE_SIZE" \
  --config_file_path "$OUT_DIR/config.json" \
  --model_repository_path "$OUT_DIR" \
  --overwrite_models

echo "=== [4/4] Done ==="
echo "Exported model is in: $OUT_DIR"
