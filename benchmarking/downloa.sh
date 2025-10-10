#!/bin/bash

# HuggingFace Dataset Downloader
# 使用镜像加速下载

echo "======================================"
echo "HuggingFace Dataset Downloader"
echo "======================================"

# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10分钟超时
export DATASETS_HTTP_TIMEOUT=600

echo "Using HF Mirror: $HF_ENDPOINT"
echo ""

# 要下载的数据集列表
DATASETS=(
    "EleutherAI/lambada_openai"
    # "Rowan/hellaswag"
    # "winogrande"
    # "allenai/ai2_arc"
)

# 下载函数
download_dataset() {
    local dataset=$1
    local max_retries=3
    
    echo "Downloading: $dataset"
    
    for i in $(seq 1 $max_retries); do
        echo "  Attempt $i/$max_retries..."
        
        python << EOF
from datasets import load_dataset
import sys

try:
    print("    Loading dataset: $dataset")
    ds = load_dataset('$dataset')
    print(f"    ✓ Success! Dataset loaded: {list(ds.keys())}")
    sys.exit(0)
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            echo "  ✓ $dataset downloaded successfully!"
            echo ""
            return 0
        else
            if [ $i -lt $max_retries ]; then
                echo "  Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "  ✗ Failed to download $dataset after $max_retries attempts"
    echo ""
    return 1
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
fi

echo "======================================"