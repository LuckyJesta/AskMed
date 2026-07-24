#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/interviewer"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${CONFIG_DIR}/interviewer_qwen3_4b_v1_lora.yaml}"
PREDICT_CONFIG="${PREDICT_CONFIG:-${CONFIG_DIR}/interviewer_qwen3_4b_v1_predict.yaml}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/outputs/interviewer_qwen3_4b/train_v1}"
TEST_OUTPUT="${TEST_OUTPUT:-${PROJECT_ROOT}/outputs/interviewer_qwen3_4b/test_v1}"
ADAPTER_DIR="${ADAPTER_DIR:-${TRAIN_OUTPUT}}"
GOLD_DATASET="${GOLD_DATASET:-${PROJECT_ROOT}/data/synthetic_interviewer/MedDG_interviewer_v1_from_v3_1_30k/MedDG_interviewer_v1_from_v3_1_30k_test_alpaca.jsonl}"
AUDIT_DATASET="${AUDIT_DATASET:-${PROJECT_ROOT}/data/synthetic_interviewer/MedDG_interviewer_v1_from_v3_1_30k/MedDG_interviewer_v1_from_v3_1_30k_validated.jsonl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: bash scripts/interviewer/finetuning/run_interviewer.sh <check|download|train|test|all> [options]

Options:
  --train-config PATH
  --predict-config PATH
  --train-output PATH
  --test-output PATH
  --adapter-dir PATH
  --gold PATH
  --audit-dataset PATH
  --model-dir PATH
  --model-id ID
EOF
}

COMMAND="${1:-}"
[[ -n "${COMMAND}" ]] || { usage; exit 1; }
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --predict-config) PREDICT_CONFIG="$2"; shift 2 ;;
    --train-output) TRAIN_OUTPUT="$2"; shift 2 ;;
    --test-output) TEST_OUTPUT="$2"; shift 2 ;;
    --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
    --gold) GOLD_DATASET="$2"; shift 2 ;;
    --audit-dataset) AUDIT_DATASET="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

abs_path() {
  [[ "$1" = /* ]] && printf "%s" "$1" || printf "%s/%s" "${PROJECT_ROOT}" "$1"
}
TRAIN_CONFIG="$(abs_path "${TRAIN_CONFIG}")"
PREDICT_CONFIG="$(abs_path "${PREDICT_CONFIG}")"
MODEL_DIR="$(abs_path "${MODEL_DIR}")"
TRAIN_OUTPUT="$(abs_path "${TRAIN_OUTPUT}")"
TEST_OUTPUT="$(abs_path "${TEST_OUTPUT}")"
ADAPTER_DIR="$(abs_path "${ADAPTER_DIR}")"
GOLD_DATASET="$(abs_path "${GOLD_DATASET}")"
AUDIT_DATASET="$(abs_path "${AUDIT_DATASET}")"
mkdir -p "${PROJECT_ROOT}/models" "${TRAIN_OUTPUT}" "${TEST_OUTPUT}"

check_environment() {
  python -m scripts.extractor.finetuning.check_environment \
    --project-root "${PROJECT_ROOT}" \
    --train-config "${TRAIN_CONFIG}" \
    --predict-config "${PREDICT_CONFIG}"
}

download_model() {
  if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model already exists: ${MODEL_DIR}"
    return
  fi
  modelscope download --model "${MODEL_ID}" --local_dir "${MODEL_DIR}"
}

train_model() {
  [[ -f "${MODEL_DIR}/config.json" ]] || { echo "Model is missing: ${MODEL_DIR}" >&2; exit 1; }
  local command=(
    llamafactory-cli train "${TRAIN_CONFIG}"
    "model_name_or_path=${MODEL_DIR}"
    "dataset_dir=${PROJECT_ROOT}/data"
    "output_dir=${TRAIN_OUTPUT}"
  )
  [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]] || command+=("resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
  "${command[@]}" 2>&1 | tee "${TRAIN_OUTPUT}/train.log"
}

test_model() {
  [[ -f "${ADAPTER_DIR}/adapter_config.json" ]] || { echo "LoRA adapter not found: ${ADAPTER_DIR}" >&2; exit 1; }
  llamafactory-cli train "${PREDICT_CONFIG}" \
    "model_name_or_path=${MODEL_DIR}" \
    "adapter_name_or_path=${ADAPTER_DIR}" \
    "dataset_dir=${PROJECT_ROOT}/data" \
    "output_dir=${TEST_OUTPUT}" \
    2>&1 | tee "${TEST_OUTPUT}/test.log"
  python -m scripts.interviewer.finetuning.evaluate_predictions \
    --predictions "${TEST_OUTPUT}/generated_predictions.jsonl" \
    --dataset "${GOLD_DATASET}" \
    --audit-dataset "${AUDIT_DATASET}" \
    --output "${TEST_OUTPUT}/structured_metrics.json" \
    2>&1 | tee "${TEST_OUTPUT}/metrics.log"
}

case "${COMMAND}" in
  check) check_environment ;;
  download) download_model ;;
  train) train_model ;;
  test) test_model ;;
  all) check_environment; download_model; train_model; test_model ;;
  *) usage; exit 1 ;;
esac
