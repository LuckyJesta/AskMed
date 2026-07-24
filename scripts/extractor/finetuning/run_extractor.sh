#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extractor"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${CONFIG_DIR}/extractor_qwen3_4b_v3_1_30k_lora.yaml}"
PREDICT_CONFIG="${PREDICT_CONFIG:-${CONFIG_DIR}/extractor_qwen3_4b_v3_1_30k_predict.yaml}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-}"
TEST_OUTPUT="${TEST_OUTPUT:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
GOLD_DATASET="${GOLD_DATASET:-}"
METRICS_OUTPUT="${METRICS_OUTPUT:-}"
TERMINOLOGY_DB="${TERMINOLOGY_DB:-}"
STANDARDIZED_PREDICTIONS_OUTPUT="${STANDARDIZED_PREDICTIONS_OUTPUT:-}"
RUN_EVAL_METRICS="${RUN_EVAL_METRICS:-true}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: bash scripts/extractor/finetuning/run_extractor.sh <command> [options]

Commands:
  check      Validate LLaMA-Factory, CUDA, BF16, configs and datasets
  download   Download ${MODEL_ID} into models/
  train      Train and validate the LoRA adapter
  test       Generate test predictions and structured metrics
  all        Run check, download, train and test

Options:
  --train-config PATH                    Training YAML, default: configs/extractor/extractor_qwen3_4b_v3_1_30k_lora.yaml
  --predict-config PATH                  Prediction YAML, default: configs/extractor/extractor_qwen3_4b_v3_1_30k_predict.yaml
  --train-output PATH                    Override training output_dir
  --test-output PATH                     Override prediction output_dir
  --adapter-dir PATH                     LoRA adapter path for test, default: train output_dir
  --gold PATH                            Gold Alpaca JSONL for metrics; default: infer from predict YAML eval_dataset
  --metrics-output PATH                  Metrics JSON path; default: <test-output>/structured_metrics*.json
  --terminology-db PATH                  Enable runtime standardization before metrics
  --standardized-predictions-output PATH Standardized prediction JSONL path
  --model-dir PATH                       Local model path, default: models/Qwen3-4B-Instruct-2507
  --model-id ID                          ModelScope model id, default: Qwen/Qwen3-4B-Instruct-2507
  --no-metrics                           Skip structured metrics after prediction

Environment variables:
  CUDA_VISIBLE_DEVICES       GPU index, default: 0
  RESUME_FROM_CHECKPOINT     Optional LLaMA-Factory checkpoint path
  TRAIN_CONFIG/PREDICT_CONFIG and most options above can also be set as env vars
EOF
}

COMMAND="${1:-}"
if [[ -z "${COMMAND}" ]]; then
  usage
  exit 1
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      TRAIN_CONFIG="$2"; shift 2 ;;
    --predict-config)
      PREDICT_CONFIG="$2"; shift 2 ;;
    --train-output)
      TRAIN_OUTPUT="$2"; shift 2 ;;
    --test-output)
      TEST_OUTPUT="$2"; shift 2 ;;
    --adapter-dir)
      ADAPTER_DIR="$2"; shift 2 ;;
    --gold)
      GOLD_DATASET="$2"; shift 2 ;;
    --metrics-output)
      METRICS_OUTPUT="$2"; shift 2 ;;
    --terminology-db)
      TERMINOLOGY_DB="$2"; shift 2 ;;
    --standardized-predictions-output)
      STANDARDIZED_PREDICTIONS_OUTPUT="$2"; shift 2 ;;
    --model-dir)
      MODEL_DIR="$2"; shift 2 ;;
    --model-id)
      MODEL_ID="$2"; shift 2 ;;
    --no-metrics)
      RUN_EVAL_METRICS="false"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf "%s" "${path}"
  else
    printf "%s/%s" "${PROJECT_ROOT}" "${path}"
  fi
}

TRAIN_CONFIG="$(abs_path "${TRAIN_CONFIG}")"
PREDICT_CONFIG="$(abs_path "${PREDICT_CONFIG}")"
MODEL_DIR="$(abs_path "${MODEL_DIR}")"
[[ -n "${TRAIN_OUTPUT}" ]] && TRAIN_OUTPUT="$(abs_path "${TRAIN_OUTPUT}")"
[[ -n "${TEST_OUTPUT}" ]] && TEST_OUTPUT="$(abs_path "${TEST_OUTPUT}")"
[[ -n "${ADAPTER_DIR}" ]] && ADAPTER_DIR="$(abs_path "${ADAPTER_DIR}")"
[[ -n "${GOLD_DATASET}" ]] && GOLD_DATASET="$(abs_path "${GOLD_DATASET}")"
[[ -n "${METRICS_OUTPUT}" ]] && METRICS_OUTPUT="$(abs_path "${METRICS_OUTPUT}")"
[[ -n "${TERMINOLOGY_DB}" ]] && TERMINOLOGY_DB="$(abs_path "${TERMINOLOGY_DB}")"
[[ -n "${STANDARDIZED_PREDICTIONS_OUTPUT}" ]] && STANDARDIZED_PREDICTIONS_OUTPUT="$(abs_path "${STANDARDIZED_PREDICTIONS_OUTPUT}")"

yaml_value() {
  local config="$1"
  local key="$2"
  python - "$config" "$key" <<'PY'
import sys
from pathlib import Path
import yaml

config = Path(sys.argv[1])
key = sys.argv[2]
data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
value = data.get(key, "")
if isinstance(value, list):
    value = ",".join(str(item) for item in value)
print(value or "")
PY
}

infer_gold_dataset() {
  local predict_config="$1"
  python - "$predict_config" "${PROJECT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
root = Path(sys.argv[2])
eval_dataset = str(config.get("eval_dataset") or config.get("dataset") or "").split(",")[0].strip()
if not eval_dataset:
    raise SystemExit("predict config has no eval_dataset")
registry = json.loads((root / "data" / "dataset_info.json").read_text(encoding="utf-8"))
row = registry.get(eval_dataset)
if not row or not row.get("file_name"):
    raise SystemExit(f"cannot infer gold dataset from registry key: {eval_dataset}")
print(root / "data" / row["file_name"])
PY
}

default_output_dir() {
  local config="$1"
  local fallback="$2"
  local output
  output="$(yaml_value "${config}" "output_dir")"
  if [[ -z "${output}" ]]; then
    printf "%s" "${fallback}"
  else
    abs_path "${output}"
  fi
}

TRAIN_OUTPUT="${TRAIN_OUTPUT:-$(default_output_dir "${TRAIN_CONFIG}" "${PROJECT_ROOT}/outputs/extractor_qwen3_4b/train")}"
TEST_OUTPUT="${TEST_OUTPUT:-$(default_output_dir "${PREDICT_CONFIG}" "${PROJECT_ROOT}/outputs/extractor_qwen3_4b/test")}"
ADAPTER_DIR="${ADAPTER_DIR:-${TRAIN_OUTPUT}}"
if [[ -z "${GOLD_DATASET}" && "${RUN_EVAL_METRICS}" == "true" ]]; then
  GOLD_DATASET="$(infer_gold_dataset "${PREDICT_CONFIG}")"
fi
if [[ -z "${METRICS_OUTPUT}" ]]; then
  if [[ -n "${TERMINOLOGY_DB}" ]]; then
    METRICS_OUTPUT="${TEST_OUTPUT}/structured_metrics_standardized.json"
  else
    METRICS_OUTPUT="${TEST_OUTPUT}/structured_metrics.json"
  fi
fi
if [[ -z "${STANDARDIZED_PREDICTIONS_OUTPUT}" && -n "${TERMINOLOGY_DB}" ]]; then
  STANDARDIZED_PREDICTIONS_OUTPUT="${TEST_OUTPUT}/generated_predictions_standardized.jsonl"
fi

mkdir -p "${PROJECT_ROOT}/models" "${TRAIN_OUTPUT}" "${TEST_OUTPUT}"

stage() {
  printf "\n[%s/4] %s\n" "$1" "$2"
}

check_environment() {
  stage 1 "Environment and dataset checks"
  local command=(
    python -m scripts.extractor.finetuning.check_environment
    --project-root "${PROJECT_ROOT}"
    --train-config "${TRAIN_CONFIG}"
    --predict-config "${PREDICT_CONFIG}"
  )
  if [[ -n "${GOLD_DATASET}" ]]; then
    command+=(--gold "${GOLD_DATASET}")
  fi
  "${command[@]}"
}

download_model() {
  stage 2 "Prepare model with ModelScope"
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
    echo "Model is missing. Run '$0 download --model-dir ${MODEL_DIR}' first." >&2
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
  "${command[@]}" 2>&1 | tee "${TRAIN_OUTPUT}/train.log"
}

test_model() {
  stage 4 "Test prediction and structured metrics"
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "LoRA adapter was not found in ${ADAPTER_DIR}." >&2
    exit 1
  fi

  llamafactory-cli train "${PREDICT_CONFIG}" \
    "model_name_or_path=${MODEL_DIR}" \
    "adapter_name_or_path=${ADAPTER_DIR}" \
    "dataset_dir=${PROJECT_ROOT}/data" \
    "output_dir=${TEST_OUTPUT}" \
    2>&1 | tee "${TEST_OUTPUT}/test.log"

  if [[ "${RUN_EVAL_METRICS}" != "true" ]]; then
    return
  fi

  local metrics_command=(
    python -m scripts.extractor.finetuning.evaluate_predictions
    --predictions "${TEST_OUTPUT}/generated_predictions.jsonl"
    --dataset "${GOLD_DATASET}"
    --output "${METRICS_OUTPUT}"
  )
  if [[ -n "${TERMINOLOGY_DB}" ]]; then
    metrics_command+=(
      --terminology-db "${TERMINOLOGY_DB}"
      --standardized-predictions-output "${STANDARDIZED_PREDICTIONS_OUTPUT}"
    )
  fi
  "${metrics_command[@]}" 2>&1 | tee "${TEST_OUTPUT}/metrics.log"
}

case "${COMMAND}" in
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
