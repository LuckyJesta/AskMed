#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/finetuning"
DATA_DIR="${PROJECT_ROOT}/data/synthetic_extractor"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/extractor_qwen3_4b"
TRAIN_OUTPUT="${OUTPUT_ROOT}/train"
TEST_OUTPUT="${OUTPUT_ROOT}/test"
TRAIN_CONFIG="${CONFIG_DIR}/extractor_qwen3_4b_lora.yaml"
PREDICT_CONFIG="${CONFIG_DIR}/extractor_qwen3_4b_predict.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${PROJECT_ROOT}/models" "${TRAIN_OUTPUT}" "${TEST_OUTPUT}"
cd "${PROJECT_ROOT}"

stage() {
  printf "\n[%s/4] %s\n" "$1" "$2"
}

check_environment() {
  stage 1 "Environment and dataset checks"
  python "${SCRIPT_DIR}/check_environment.py" --project-root "${PROJECT_ROOT}"
}

download_model() {
  stage 2 "Prepare Qwen3-4B-Instruct-2507 with ModelScope"
  if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model already exists: ${MODEL_DIR}"
    return
  fi

  if ! command -v modelscope >/dev/null 2>&1; then
    echo "modelscope CLI was not found in the active environment." >&2
    exit 1
  fi

  modelscope download \
    --model "${MODEL_ID}" \
    --local_dir "${MODEL_DIR}"
}

train_model() {
  stage 3 "LoRA training and validation"
  if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model is missing. Run '$0 download' first." >&2
    exit 1
  fi

  local command=(
    llamafactory-cli train "${TRAIN_CONFIG}"
    "model_name_or_path=${MODEL_DIR}"
    "dataset_dir=${PROJECT_ROOT}/data"
    "output_dir=${TRAIN_OUTPUT}"
  )
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    command+=("resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
  fi

  printf "Command:"
  printf " %q" "${command[@]}"
  printf "\n"
  "${command[@]}" 2>&1 | tee "${OUTPUT_ROOT}/train.log"
}

test_model() {
  stage 4 "Test prediction and structured metrics"
  if [[ ! -f "${TRAIN_OUTPUT}/adapter_config.json" ]]; then
    echo "LoRA adapter was not found in ${TRAIN_OUTPUT}." >&2
    exit 1
  fi

  llamafactory-cli train "${PREDICT_CONFIG}" \
    "model_name_or_path=${MODEL_DIR}" \
    "adapter_name_or_path=${TRAIN_OUTPUT}" \
    "dataset_dir=${PROJECT_ROOT}/data" \
    "output_dir=${TEST_OUTPUT}" \
    2>&1 | tee "${OUTPUT_ROOT}/test.log"

  python "${SCRIPT_DIR}/evaluate_predictions.py" \
    --predictions "${TEST_OUTPUT}/generated_predictions.jsonl" \
    --dataset "${DATA_DIR}/MedDG_extractor_15k_test_alpaca.jsonl" \
    --output "${TEST_OUTPUT}/structured_metrics.json" \
    2>&1 | tee "${OUTPUT_ROOT}/metrics.log"
}

usage() {
  cat <<EOF
Usage: bash scripts/finetuning/run_extractor.sh <command>

Commands:
  check      Validate LLaMA-Factory, CUDA, BF16 and datasets
  download   Download ${MODEL_ID} into models/
  train      Train and validate the LoRA adapter
  test       Generate test predictions and structured metrics
  all        Run check, download, train and test

Environment variables:
  CUDA_VISIBLE_DEVICES       GPU index, default: 0
  RESUME_FROM_CHECKPOINT     Optional LLaMA-Factory checkpoint path
  PYTORCH_CUDA_ALLOC_CONF    Optional allocator settings supported by the installed PyTorch version
EOF
}

case "${1:-}" in
  check)
    check_environment
    ;;
  download)
    download_model
    ;;
  train)
    train_model
    ;;
  test)
    test_model
    ;;
  all)
    check_environment
    download_model
    train_model
    test_model
    ;;
  *)
    usage
    exit 1
    ;;
esac
