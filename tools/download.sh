#!/bin/bash

# HuggingFace Dataset Downloader

echo "======================================"
echo "HuggingFace Dataset Downloader"
echo "======================================"
echo ""

# 配置环境变量
export HF_TOKEN=""
# 对于需要认证的数据集，不使用镜像，直接用官方源
unset HF_ENDPOINT
export HF_HUB_DOWNLOAD_TIMEOUT=600
export DATASETS_HTTP_TIMEOUT=600

echo "Configuration:"
echo "  HF_ENDPOINT: ${HF_ENDPOINT:-https://huggingface.co (official)}"
echo "  HF_TOKEN: ${HF_TOKEN:0:10}..."
echo ""

# 要下载的数据集列表
DATASETS=(
    "TIGER-Lab/MMLU-Pro"
    # "cais/mmlu"
    # "Rowan/hellaswag"
    # "winogrande"
    # "allenai/ai2_arc"
)

# 下载单个数据集
download_dataset() {
    local dataset=$1
    echo "Downloading: $dataset"
    
    python << EOF
from datasets import load_dataset
import sys
import os

try:
    # 使用 token 进行认证（MMLU-Pro 需要登录）
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('$dataset', token=token)
    print(f"✓ Success! Dataset loaded: {list(ds.keys())}")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        return 0
    else
        echo ""
        return 1
    fi
}

# 下载所有数据集
failed_datasets=()

for dataset in "${DATASETS[@]}"; do
    download_dataset "$dataset"
    if [ $? -ne 0 ]; then
        failed_datasets+=("$dataset")
    fi
done

# 总结
echo "======================================"
echo "Download Summary"
echo "======================================"

if [ ${#failed_datasets[@]} -eq 0 ]; then
    echo "✓ All datasets downloaded successfully!"
else
    echo "✗ Failed datasets:"
    for dataset in "${failed_datasets[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Troubleshooting:"
    echo "  1. Try official HuggingFace: unset HF_ENDPOINT"
    echo "  2. Check network: ping hf-mirror.com"
    echo "  3. Verify token: echo \$HF_TOKEN"
fi

echo "======================================"