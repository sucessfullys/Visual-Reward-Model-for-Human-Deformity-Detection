#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${1:-$SCRIPT_DIR/hand_yolox_pai_640.onnx}"
ENGINE_PATH="${2:-$SCRIPT_DIR/hand_yolox_pai_640_fp16.engine}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec was not found. Install TensorRT for CUDA 12.9 first." >&2
  exit 1
fi

trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --shapes=images:1x3x640x640 \
  --skipInference

echo "TensorRT engine created: $ENGINE_PATH"
