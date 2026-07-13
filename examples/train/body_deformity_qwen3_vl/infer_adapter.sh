#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:?Please set ADAPTER_PATH to a saved LoRA checkpoint directory.}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/sample_body_deformity.jsonl}"
RESULT_PATH="${RESULT_PATH:-${PROJECT_DIR}/output/body_deformity_qwen3_vl_infer.jsonl}"

require_abs_path() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != /* ]]; then
        echo "Error: ${name} must be an absolute path, got: ${value}" >&2
        exit 1
    fi
}

require_abs_path MODEL_PATH "${MODEL_PATH}"
require_abs_path ADAPTER_PATH "${ADAPTER_PATH}"
require_abs_path VAL_DATASET_PATH "${VAL_DATASET_PATH}"
require_abs_path RESULT_PATH "${RESULT_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-512}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-64}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-8}"

if command -v swift >/dev/null 2>&1; then
    SWIFT_CMD=(swift)
else
    SWIFT_CMD=(python -m swift.cli.main)
fi

"${SWIFT_CMD[@]}" infer \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --adapters "${ADAPTER_PATH}" \
    --stream false \
    --infer_backend transformers \
    --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --val_dataset_sample "${VAL_DATASET_SAMPLE:-1}" \
    --result_path "${RESULT_PATH}"
