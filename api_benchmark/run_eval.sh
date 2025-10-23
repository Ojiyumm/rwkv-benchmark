#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OPENAI_API_KEY="EMPTY"

declare -A TASK_FREESHOT_OVERRIDES=(
  ["hendrycks_math"]="--num_fewshot 0"
  ["gpqa_main_zeroshot"]="--num_fewshot 0"
  ["ifeval"]="--num_fewshot 0"
)

TASKS=(
  "aime24"
  "aime25"
  "gpqa_main_zeroshot"
  "mmlu"
  "mmlu_pro"
  "mmlu_pro_plus"
  "mmlu_prox_en"
#  "humaneval"
#  "mbpp"
  "asdiv"
  "gsm_plus"
  "gsm8k"
  "ifeval"
  "ruler"
  "hendrycks_math"
  "minerva_math"
)

for TASK in "${TASKS[@]}"; do
  echo ">>> Running ${TASK}"
  FEWSHOT_OVERRIDE="${TASK_FREESHOT_OVERRIDES[$TASK]:-}"
  lm_eval \
    --model openai-completions \
    --model_args base_url="http://10.100.1.98:16384/v1/completions",model=davinci-002,tokenized_requests=false \
    --tasks "${TASK}" \
    --output_path ./eval_results \
    --batch_size auto \
    "$@"
done
