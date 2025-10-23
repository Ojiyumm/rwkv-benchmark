#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=900
export DATASETS_HTTP_TIMEOUT=900
export HF_TOKEN=

echo "======================================"
echo "HuggingFace Dataset Downloader"
echo "Using endpoint: ${HF_ENDPOINT}"
echo "======================================"

# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10分钟超时
export DATASETS_HTTP_TIMEOUT=600

echo "Using HF Mirror: $HF_ENDPOINT"
echo ""

# 要下载的数据集列表
DATASETS=(
  "Maxwell-Jia/AIME_2024::"
  "math-ai/aime25::"
  "Idavidrein/gpqa::"                     # gated, requires HF_TOKEN + main endpoint
  "cais/mmlu::all"
  "TIGER-Lab/MMLU-Pro::"
  "saeidasgari/mmlu-pro-plus::"
  "li-lab/MMLU-ProX::en"
  "openai/openai_humaneval::"
  "google-research-datasets/mbpp::full"
  "EleutherAI/asdiv::"
  "qintongli/GSM-Plus::"
  "gsm8k::main"
  "google/IFEval::"
  "EleutherAI/hendrycks_math::algebra"
  "EleutherAI/hendrycks_math::counting_and_probability"
  "EleutherAI/hendrycks_math::geometry"
  "EleutherAI/hendrycks_math::intermediate_algebra"
  "EleutherAI/hendrycks_math::number_theory"
  "EleutherAI/hendrycks_math::prealgebra"
  "EleutherAI/hendrycks_math::precalculus"
)

BENCHMARK_DIR="${SCRIPT_DIR}/../benchmarkdata"
mkdir -p "${BENCHMARK_DIR}"

declare -A OUTPUT_MAP=(
  ["Maxwell-Jia/AIME_2024::"]="aime24"
  ["math-ai/aime25::"]="aime25"
  ["Idavidrein/gpqa::"]="gpqa"
  ["cais/mmlu::all"]="mmlu"
  ["TIGER-Lab/MMLU-Pro::"]="mmlupro"
  ["saeidasgari/mmlu-pro-plus::"]="mmlu_pro_plus"
  ["li-lab/MMLU-ProX::en"]="mmlu_prox/en"
  ["openai/openai_humaneval::"]="humaneval"
  ["google-research-datasets/mbpp::full"]="mbpp"
  ["EleutherAI/asdiv::"]="asdiv"
  ["qintongli/GSM-Plus::"]="gsm_plus"
  ["gsm8k::main"]="gsm8k"
  ["google/IFEval::"]="ifeval"
  ["EleutherAI/hendrycks_math::algebra"]="hendrycks_math/algebra"
  ["EleutherAI/hendrycks_math::counting_and_probability"]="hendrycks_math/counting_and_probability"
  ["EleutherAI/hendrycks_math::geometry"]="hendrycks_math/geometry"
  ["EleutherAI/hendrycks_math::intermediate_algebra"]="hendrycks_math/intermediate_algebra"
  ["EleutherAI/hendrycks_math::number_theory"]="hendrycks_math/number_theory"
  ["EleutherAI/hendrycks_math::prealgebra"]="hendrycks_math/prealgebra"
  ["EleutherAI/hendrycks_math::precalculus"]="hendrycks_math/precalculus"
)

declare -A FORMAT_MAP=(
  ["TIGER-Lab/MMLU-Pro::"]="parquet"
)

download_dataset() {
  local dataset_spec="$1"
  local dataset_id="${dataset_spec%%::*}"
  local dataset_config="${dataset_spec##*::}"

  [[ "${dataset_config}" == "${dataset_id}" ]] && dataset_config=""

  local target_rel="${OUTPUT_MAP[$dataset_spec]}"
  if [[ -z "${target_rel}" ]]; then
    target_rel="${dataset_spec//\//_}"
    target_rel="${target_rel//:/_}"
  fi
  local target_dir="${BENCHMARK_DIR}/${target_rel}"
  local format="${FORMAT_MAP[$dataset_spec]:-json}"

  echo "Downloading ${dataset_id}${dataset_config:+ (config=${dataset_config})}"
  echo "  → saving to ${target_dir} (format=${format})"

  rm -rf "${target_dir}"
  mkdir -p "${target_dir}"

  python3 - "$dataset_id" "$dataset_config" "$target_dir" "$format" <<'PYCODE'
import os
import sys
from datasets import load_dataset

dataset_id = sys.argv[1]
dataset_config = sys.argv[2] or None
target_dir = sys.argv[3]
file_format = sys.argv[4]

dataset = load_dataset(dataset_id, dataset_config, trust_remote_code=True)

if hasattr(dataset, "keys"):
    dataset_dict = dataset
else:
    dataset_dict = {"train": dataset}

saved = []
for split, subset in dataset_dict.items():
    if file_format == "parquet":
        out_path = os.path.join(target_dir, f"{split}-00000-of-00001.parquet")
        subset.to_parquet(out_path)
    else:
        out_path = os.path.join(target_dir, f"{split}.jsonl")
        subset.to_json(out_path, orient="records", lines=True, force_ascii=False)
    saved.append((split, out_path))

print(f"  ✓ cached splits: {[split for split, _ in saved]}")
for split, out_path in saved:
    print(f"    - {split}: {out_path}")
PYCODE
}

failed=()

for spec in "${DATASETS[@]}"; do
  if ! download_dataset "${spec}"; then
    failed+=("${spec}")
    echo "  retrying in 5 seconds..."
    sleep 5
    download_dataset "${spec}" || true
  fi
  echo ""
done

if ((${#failed[@]})); then
  echo "======================================"
  echo "Download Summary: failures detected"
  printf '  - %s\n' "${failed[@]}"
  echo "======================================"
  exit 1
fi

echo "======================================"
echo "All datasets cached successfully."
echo "======================================"
