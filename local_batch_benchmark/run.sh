#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PIPELINE="${SCRIPT_DIR}/run_pipeline.py"

MODELS_DIR="/public/home/ssjxzkz/Weights/BlinkDL__rwkv7-g1"
DATA_ROOT="${PROJECT_ROOT}/../benchmarkdata"

if [[ ! -x "${PIPELINE}" ]]; then
  echo "run_pipeline.py not found at ${PIPELINE}" >&2
  exit 1
fi

shopt -s nullglob
MODEL_FILES=("${MODELS_DIR}"/*.pth)
shopt -u nullglob

if ((${#MODEL_FILES[@]} == 0)); then
  echo "No .pth models found under ${MODELS_DIR}" >&2
  exit 1
fi

source ../.venv/bin/activate

DATASET_ORDER=(
#  "aime24"
#  "aime25"
  "gpqa_main_zeroshot"
  "mmlu"
  "mmlu_pro"
  "mmlu_pro_plus"
  "mmlu_prox"
  "asdiv"
  "gsm_plus"
  "gsm8k"
  "ifeval"
  "ruler"
  "hendrycks_math"

#  "minerva_math"
)

declare -A DATASET_PATHS=(
  ["aime24"]="${DATA_ROOT}/aime24/train.jsonl"
  ["aime25"]="${DATA_ROOT}/aime25/test.jsonl"
  ["gpqa_main_zeroshot"]="${DATA_ROOT}/gpqa/train.jsonl"
  ["mmlu"]="${DATA_ROOT}/mmlu/test.jsonl"
  ["mmlu_pro"]="${DATA_ROOT}/mmlupro"
  ["mmlu_pro_plus"]="${DATA_ROOT}/mmlu_pro_plus/test.jsonl"
  ["mmlu_prox"]="${DATA_ROOT}/mmlu_prox/en/test.jsonl"
  ["asdiv"]="${DATA_ROOT}/asdiv/validation.jsonl"
  ["gsm_plus"]="${DATA_ROOT}/gsm_plus/test.jsonl"
  ["gsm8k"]="${DATA_ROOT}/gsm8k/test.jsonl"
  ["ifeval"]="${DATA_ROOT}/ifeval/train.jsonl"
  ["ruler"]="${DATA_ROOT}/ruler/train.jsonl"
  ["hendrycks_math"]="${DATA_ROOT}/hendrycks_math/*/test.jsonl"
#  ["minerva_math"]="${DATA_ROOT}/minerva_math/*/test.jsonl"
)

declare -A EVALUATORS=(
  ["aime24"]="generation"
  ["aime25"]="generation"
  ["gpqa_main_zeroshot"]="mmlu_pro"
  ["mmlu"]="mmlu_pro"
  ["mmlu_pro"]="mmlu_pro"
  ["mmlu_pro_plus"]="mmlu_pro"
  ["mmlu_prox"]="mmlu_pro"
  ["asdiv"]="exact_match"
  ["gsm_plus"]="exact_match"
  ["gsm8k"]="exact_match"
  ["ifeval"]="generation"
  ["ruler"]="generation"
  ["hendrycks_math"]="generation"
#  ["minerva_math"]="generation"
)

echo "Using models in ${MODELS_DIR}"
echo "Datasets will be read from ${DATA_ROOT}"
echo ""

for model_path in "${MODEL_FILES[@]}"; do
  model_name="$(basename "${model_path}" .pth)"
  # Strip .pth extension for the Python script since rwkv7.py adds it automatically
  model_path_for_python="${model_path%.pth}"
  echo "==============================="
  echo "Model: ${model_name}"
  echo "==============================="

  for dataset in "${DATASET_ORDER[@]}"; do
    dataset_path="${DATASET_PATHS[$dataset]}"
    evaluator="${EVALUATORS[$dataset]}"

    if [[ -z "${dataset_path:-}" || -z "${evaluator:-}" ]]; then
      echo "  [${dataset}] Missing dataset path or evaluator, skipping."
      continue
    fi

    if [[ "${dataset_path}" != *"*"* && ! -e "${dataset_path}" ]]; then
      echo "  [${dataset}] Dataset path not found: ${dataset_path}, skipping."
      continue
    fi

    output_dir="${SCRIPT_DIR}/eval_results/${model_name}/${dataset}"
    mkdir -p "${output_dir}"

    echo "  [${dataset}] evaluator=${evaluator}"
    uv run "${PIPELINE}" \
      --model-path "${model_path_for_python}" \
      --dataset "${dataset}" \
      --dataset-path "${dataset_path}" \
      --evaluator "${evaluator}" \
      --output-dir "${output_dir}" \
      --batch-size 80 \
#      --max-length 16384
  done

  echo ""
done

echo "All evaluations completed."
