#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_DIR}/examples/train/body_deformity_qwen3_vl/sample_body_deformity.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/body_deformity_qwen3_vl_smoke}"

require_abs_path() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != /* ]]; then
        echo "Error: ${name} must be an absolute path, got: ${value}" >&2
        exit 1
    fi
}

require_abs_path MODEL_PATH "${MODEL_PATH}"
require_abs_path DATASET_PATH "${DATASET_PATH}"
require_abs_path OUTPUT_DIR "${OUTPUT_DIR}"

python - <<'PY'
import importlib.metadata as metadata
from packaging.version import Version

try:
    version = metadata.version("qwen-vl-utils")
except metadata.PackageNotFoundError:
    raise SystemExit(
        "Missing dependency: qwen-vl-utils>=0.0.14\n"
        "Install it in the active ms-swift environment:\n"
        "  python -m pip install -U 'qwen-vl-utils>=0.0.14' decord\n"
    )

if Version(version) < Version("0.0.14"):
    raise SystemExit(
        f"qwen-vl-utils is too old: {version}. Required: >=0.0.14\n"
        "Upgrade it in the active ms-swift environment:\n"
        "  python -m pip install -U 'qwen-vl-utils>=0.0.14' decord\n"
    )
PY

MODEL_PATH_FOR_CHECK="${MODEL_PATH}" python - <<'PY'
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_PATH_FOR_CHECK"])
shards = sorted(model_dir.glob("*.safetensors"))
if not shards:
    raise SystemExit(f"No .safetensors files found in MODEL_PATH: {model_dir}")

bad = []
for path in shards:
    try:
        head = path.read_bytes()[:80]
    except OSError as exc:
        bad.append(f"{path}: cannot read file ({exc})")
        continue
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        bad.append(f"{path}: Git LFS pointer, real weights were not downloaded")
    elif path.stat().st_size < 1024 * 1024:
        bad.append(f"{path}: suspiciously small ({path.stat().st_size} bytes)")

if bad:
    msg = "\n".join(bad)
    raise SystemExit(
        "Invalid model weight files detected:\n"
        f"{msg}\n\n"
        "Fix the model directory before training, for example:\n"
        f"  cd {model_dir}\n"
        "  git lfs install\n"
        "  git lfs pull\n"
    )
PY

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

DS_ARGS=()
if [[ -n "${DEEPSPEED:-}" ]]; then
    DS_ARGS=(--deepspeed "${DEEPSPEED}")
fi

"${SWIFT_CMD[@]}" sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --dataset "${DATASET_PATH}" \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
    --warmup_ratio "${WARMUP_RATIO:-0.05}" \
    --weight_decay "${WEIGHT_DECAY:-0.1}" \
    --adam_beta1 "${ADAM_BETA1:-0.9}" \
    --adam_beta2 "${ADAM_BETA2:-0.95}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --lora_rank "${LORA_RANK:-8}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-1}" \
    --save_total_limit 1 \
    --eval_strategy "${EVAL_STRATEGY:-no}" \
    --eval_steps "${EVAL_STEPS:-${SAVE_STEPS:-1}}" \
    --logging_steps 1 \
    --max_length "${MAX_LENGTH:-2048}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED:-42}" \
    --data_seed "${DATA_SEED:-42}" \
    --dataset_shuffle "${DATASET_SHUFFLE:-true}" \
    --dataset_num_proc 1 \
    --dataloader_num_workers 1 \
    --save_only_model true \
    "${DS_ARGS[@]}"
