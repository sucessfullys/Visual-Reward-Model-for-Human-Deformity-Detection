#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-XXX/python}"
SAM3_DIR="XXX/sam3"
SCRIPT="${SAM3_DIR}/classify_aigi_real_person_sam3.py"

INPUT_DIR="${INPUT_DIR:-XXX}"
OUTPUT_ROOT="${OUTPUT_ROOT:-XXX}"
HUMAN_IMAGE_DIR="${HUMAN_IMAGE_DIR:-${OUTPUT_ROOT}/real_human/images}"
HUMAN_LABEL_DIR="${HUMAN_LABEL_DIR:-${OUTPUT_ROOT}/real_human/labels}"
NO_HUMAN_DIR="${NO_HUMAN_DIR:-${OUTPUT_ROOT}/real_no_human}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-XXX/sam3/hf_sam3.1/sam3.1_multiplex.pt}"
PROMPT="${PROMPT:-person}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
SHARD_COUNT="${SHARD_COUNT:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
LIMIT="${LIMIT:-0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/real_human_sam3_logs}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${HUMAN_IMAGE_DIR}" "${HUMAN_LABEL_DIR}" "${NO_HUMAN_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Error: SHARD_COUNT=${SHARD_COUNT}, but GPU_IDS has ${#GPUS[@]} entries: ${GPU_IDS}" >&2
  exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -V 2>&1)"
echo "Input: ${INPUT_DIR}"
echo "Human images: ${HUMAN_IMAGE_DIR}"
echo "Human labels: ${HUMAN_LABEL_DIR}"
echo "No human: ${NO_HUMAN_DIR}"
echo "Logs: ${LOG_DIR}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  log_file="${LOG_DIR}/shard_${shard_index}.log"
  json_log="${LOG_DIR}/shard_${shard_index}.jsonl"
  resume_flag=()
  if [[ "${RESUME}" == "true" ]]; then
    resume_flag=(--resume)
  fi
  echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_file}"
  (
    cd "${SAM3_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
      --input-dir "${INPUT_DIR}" \
      --human-image-dir "${HUMAN_IMAGE_DIR}" \
      --human-label-dir "${HUMAN_LABEL_DIR}" \
      --no-human-dir "${NO_HUMAN_DIR}" \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --prompt "${PROMPT}" \
      --threshold "${THRESHOLD}" \
      --max-objects "${MAX_OBJECTS}" \
      --device "cuda:0" \
      --shard-index "${shard_index}" \
      --shard-count "${SHARD_COUNT}" \
      --limit "${LIMIT}" \
      --log-jsonl "${json_log}" \
      "${resume_flag[@]}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

echo "Shard jobs finished. failed=${failed}"
echo "human_images=$(find "${HUMAN_IMAGE_DIR}" -maxdepth 1 -type f | wc -l)"
echo "human_labels=$(find "${HUMAN_LABEL_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
echo "no_human_images=$(find "${NO_HUMAN_DIR}" -maxdepth 1 -type f | wc -l)"

exit "${failed}"
