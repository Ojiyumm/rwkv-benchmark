#!/bin/bash

# 自动将 RWKV 批量推理集成到 nemo-skills
# Usage: bash install_to_nemo_skills.sh

set -e

NEMO_SKILLS_DIR="/home/rwkv/peter/Skills"
RWKV_API_DIR="/home/rwkv/peter/rwkv-benchmark/api_benchmark"

echo "=========================================="
echo "RWKV Batch Inference -> Nemo-Skills"
echo "=========================================="
echo ""

# 检查目录是否存在
if [ ! -d "$NEMO_SKILLS_DIR" ]; then
    echo "❌ 错误: nemo-skills 目录不存在: $NEMO_SKILLS_DIR"
    exit 1
fi

if [ ! -f "$RWKV_API_DIR/nemo_skills_rwkv_model.py" ]; then
    echo "❌ 错误: RWKV 模型文件不存在: $RWKV_API_DIR/nemo_skills_rwkv_model.py"
    exit 1
fi

echo "1️⃣  复制 RWKV 模型文件..."
cp "$RWKV_API_DIR/nemo_skills_rwkv_model.py" \
   "$NEMO_SKILLS_DIR/nemo_skills/inference/model/rwkv_batch.py"
echo "   ✅ 已复制到: $NEMO_SKILLS_DIR/nemo_skills/inference/model/rwkv_batch.py"

echo ""
echo "2️⃣  注册 RWKV 模型到 nemo-skills..."

INIT_FILE="$NEMO_SKILLS_DIR/nemo_skills/inference/model/__init__.py"
BACKUP_FILE="$INIT_FILE.backup"

# 备份原文件
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$INIT_FILE" "$BACKUP_FILE"
    echo "   📦 已备份原文件: $BACKUP_FILE"
fi

# 检查是否已经添加过
if grep -q "rwkv_batch" "$INIT_FILE"; then
    echo "   ⚠️  RWKV 模型已注册，跳过"
else
    # 在 import 部分添加
    if grep -q "from .vllm import VLLMModel" "$INIT_FILE"; then
        sed -i '/from .vllm import VLLMModel/a from .rwkv_batch import RWKVBatchModel, RWKVDirectModel' "$INIT_FILE"
        echo "   ✅ 已添加 import"
    else
        echo "   ⚠️  未找到插入点，请手动添加 import"
    fi
    
    # 在 models 字典中添加
    if grep -q '"vllm": VLLMModel,' "$INIT_FILE"; then
        sed -i '/"vllm": VLLMModel,/a \    "rwkv_batch": RWKVBatchModel,\n    "rwkv_direct": RWKVDirectModel,' "$INIT_FILE"
        echo "   ✅ 已注册到 models 字典"
    else
        echo "   ⚠️  未找到插入点，请手动添加到 models 字典"
    fi
fi

echo ""
echo "3️⃣  验证安装..."

if python -c "from nemo_skills.inference.model.rwkv_batch import RWKVBatchModel; print('Import OK')" 2>/dev/null; then
    echo "   ✅ RWKV 模型导入成功"
else
    echo "   ❌ RWKV 模型导入失败，请检查安装"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "现在可以使用以下模式："
echo ""
echo "  1. Batch Accumulator（推荐）："
echo "     ++server.server_type=rwkv_batch"
echo ""
echo "  2. Direct API："
echo "     ++server.server_type=rwkv_direct"
echo ""
echo "详细使用说明："
echo "  cat $RWKV_API_DIR/NEMO_SKILLS_INTEGRATION.md"
echo ""
echo "回滚安装："
echo "  cp $BACKUP_FILE $INIT_FILE"
echo ""

