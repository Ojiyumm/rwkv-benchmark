#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/ssjxzkz/Projects/rwkv-benchmark
WEIGHT_ROOT=/public/home/ssjxzkz/Weights/BlinkDL__rwkv7-g1
OUT_BASE=$ROOT/results/nemo-skills
VOCAB=$ROOT/reference/rwkv_vocab_v20230424.txt
# 全量可用 benchmark（目录下所有数据集模块）
# 跑的慢 mmlu-prox
# 跑不通 scicode,
# 缺数据 livecodebench,livebench-coding,livecodebench-pro,mrcr
#BENCHMARKS="hle,aai,aalcr,aime24,aime25,algebra222,amc23,answer-judge,arena-hard,asdiv,beyond-aime,bfcl_v3,bigcodebench,brumo25,college_math,comp-math-24-25,flores200,gaokao2023en,gpqa,gsm-plus,gsm8k,hmmt_feb25,ifbench,ifeval,ioi24,svamp,wmt24pp"
BENCHMARKS="gsm8k,aime24,aime25,gsm-plus,math-odyssey,hmmt_feb25,mmlu-pro,livecodebench,svamp,wmt24pp"
#BENCHMARKS="math,math-500,math-odyssey,mawps,mbpp,minerva_math,minif2f,mmlu,mmlu-pro,mmlu-redux,mobench,ojbench,olympiadbench,omni-math,proofnet,putnam-bench,ruler,simpleqa,supergpqa,svamp,swe-bench,wmt24pp"
COMMON_OVERRIDES="++server.vocab_path=${VOCAB} ++server.max_batch_size=160 ++inference.temperature=0.0 ++inference.tokens_to_generate=2048 ++max_concurrent_requests=40"
DATA_DIR=$ROOT/nemo-skills/nemo_skills/dataset

# Use environment variable so `ns eval` and related tooling pick up the same data_dir.
export NEMO_SKILLS_DATA_DIR="$DATA_DIR"
export HF_TOKEN=


# Derive the dataset list from BENCHMARKS (supporting suffixes like `:4` or `.llama_128k`).
IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"
declare -A SEEN_DATASETS=()
DATASETS_TO_PREPARE=()
for RAW_BENCH in "${BENCH_LIST[@]}"; do
    CLEAN_BENCH=${RAW_BENCH//[$'\t\r\n ']/}
    [[ -z "$CLEAN_BENCH" ]] && continue
    DATASET=${CLEAN_BENCH%%:*}
    [[ -z "$DATASET" ]] && continue
    if [[ -z "${SEEN_DATASETS[$DATASET]:-}" ]]; then
        DATASETS_TO_PREPARE+=("$DATASET")
        SEEN_DATASETS[$DATASET]=1
    fi
done

# Ensure evaluation data is present before launching any jobs. `ns prepare_data`
# is idempotent, so re-running the script simply refreshes the cached datasets.
#ns prepare_data --cluster=local-nodocker --data_dir="$DATA_DIR" "${DATASETS_TO_PREPARE[@]}"
#
## Ensure no stale NeMo-Run / TorchX processes or state remain.
pkill -f "nemo_skills\.code_execution\.local_sandbox" 2>/dev/null || true
pkill -f "ns eval" 2>/dev/null || true
pkill -f "torchx" 2>/dev/null || true
rm -rf ~/.nemo_run ~/.torchx

export PYTHONPATH=$ROOT

find "$WEIGHT_ROOT" -type f -name '*.pth' -print0 | sort -z | \
while IFS= read -r -d '' CKPT; do
    PREFIX=${CKPT%.pth}
    NAME=$(basename "$PREFIX")
    OUTDIR=${OUT_BASE}/${NAME}

    IFS=',' read -ra BENCH_ARRAY <<< "$BENCHMARKS"
    for BENCH in "${BENCH_ARRAY[@]}"; do
        ns eval \
          --cluster=local-nodocker \
          --server_type=rwkv_local \
          --model="$PREFIX" \
          --benchmarks="$BENCH" \
          --output_dir="${OUTDIR}/${BENCH}" \
          --expname="${NAME}-${BENCH}" \
          --single_node_mode=sequential \
          "$COMMON_OVERRIDES"
    done

    ns summarize_results --cluster=local-nodocker "$OUTDIR"
done
