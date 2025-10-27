#!/bin/bash

# Nemo-Skills + RWKV 完整示例
# 展示如何使用三种模式进行评估

set -e

# ============================================
# 配置
# ============================================

NEMO_SKILLS_DIR="/home/rwkv/Peter/Skills"
DATA_DIR="/home/rwkv/Peter/Skills/datasets"
OUTPUT_BASE="/home/rwkv/Peter/Skills/outputs/rwkv_test"

# API 服务器配置
API_HOST="192.168.0.82"
API_PORT="8001"
MODEL_NAME="rwkv-7-world"

# 测试数据（使用少量数据快速测试）
TEST_DATA="$DATA_DIR/gsm8k/test_small.jsonl"  # 你需要准备这个文件

# 如果测试数据不存在，创建一个小样本
if [ ! -f "$TEST_DATA" ]; then
    mkdir -p "$(dirname $TEST_DATA)"
    echo "创建测试数据..."
    cat > "$TEST_DATA" << 'EOF'
{"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "answer": "9"}
{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"}
{"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"}
EOF
    echo "✅ 已创建测试数据: $TEST_DATA"
fi

cd "$NEMO_SKILLS_DIR"

# ============================================
# 示例 1: Batch Accumulator（推荐，最快）
# ============================================

echo ""
echo "=========================================="
echo "示例 1: Batch Accumulator 模式"
echo "=========================================="
echo ""

OUTPUT_DIR="$OUTPUT_BASE/batch_accumulator"
mkdir -p "$OUTPUT_DIR"

echo "配置:"
echo "  - server_type: rwkv_batch"
echo "  - batch_size: 128"
echo "  - max_concurrent_requests: 256"
echo ""

ns generate \
    ++input_file="$TEST_DATA" \
    ++output_dir="$OUTPUT_DIR" \
    ++prompt_config=gsm8k/base \
    ++server.server_type=rwkv_batch \
    ++server.host="$API_HOST" \
    ++server.port="$API_PORT" \
    ++server.model="$MODEL_NAME" \
    ++server.batch_size=128 \
    ++server.max_wait_ms=50 \
    ++inference.tokens_to_generate=512 \
    ++inference.temperature=0.0 \
    ++max_concurrent_requests=256

echo ""
echo "✅ Batch Accumulator 完成"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# ============================================
# 示例 2: Dynamic Batching（兼容模式）
# ============================================

echo ""
echo "=========================================="
echo "示例 2: Dynamic Batching 模式"
echo "=========================================="
echo ""
echo "注意: 需要服务器启动时设置 --batch_mode dynamic"
echo "      当前服务器模式，请检查："
curl -s "http://$API_HOST:$API_PORT/stats" | grep batch_mode || echo "  无法获取"
echo ""

OUTPUT_DIR="$OUTPUT_BASE/dynamic_batching"
mkdir -p "$OUTPUT_DIR"

echo "配置:"
echo "  - server_type: openai (兼容模式)"
echo "  - 服务器自动累积批次"
echo ""

ns generate \
    ++input_file="$TEST_DATA" \
    ++output_dir="$OUTPUT_DIR" \
    ++prompt_config=gsm8k/base \
    ++server.server_type=openai \
    ++server.base_url="http://$API_HOST:$API_PORT/v1" \
    ++server.model="$MODEL_NAME" \
    ++inference.tokens_to_generate=512 \
    ++inference.temperature=0.0 \
    ++max_concurrent_requests=256

echo ""
echo "✅ Dynamic Batching 完成"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# ============================================
# 示例 3: Direct API（基准对比）
# ============================================

echo ""
echo "=========================================="
echo "示例 3: Direct API 模式（基准）"
echo "=========================================="
echo ""

OUTPUT_DIR="$OUTPUT_BASE/direct"
mkdir -p "$OUTPUT_DIR"

echo "配置:"
echo "  - server_type: rwkv_direct"
echo "  - 每次推理一个 prompt"
echo ""

ns generate \
    ++input_file="$TEST_DATA" \
    ++output_dir="$OUTPUT_DIR" \
    ++prompt_config=gsm8k/base \
    ++server.server_type=rwkv_direct \
    ++server.host="$API_HOST" \
    ++server.port="$API_PORT" \
    ++server.model="$MODEL_NAME" \
    ++inference.tokens_to_generate=512 \
    ++inference.temperature=0.0 \
    ++max_concurrent_requests=16

echo ""
echo "✅ Direct API 完成"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# ============================================
# 性能对比
# ============================================

echo ""
echo "=========================================="
echo "性能对比"
echo "=========================================="
echo ""

for mode in batch_accumulator dynamic_batching direct; do
    output_file="$OUTPUT_BASE/$mode/output-rs0.jsonl"
    if [ -f "$output_file" ]; then
        count=$(wc -l < "$output_file")
        
        # 提取平均生成时间
        avg_time=$(python3 - <<EOF
import json
total_time = 0
count = 0
with open("$output_file") as f:
    for line in f:
        data = json.loads(line)
        if "generation_time" in data:
            total_time += data["generation_time"]
            count += 1
if count > 0:
    print(f"{total_time/count:.3f}")
else:
    print("N/A")
EOF
)
        
        echo "$mode:"
        echo "  样本数: $count"
        echo "  平均时间: ${avg_time}s/sample"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  ls -lh $OUTPUT_BASE/*/output-rs0.jsonl"
echo ""
echo "查看详细输出:"
echo "  head -n 1 $OUTPUT_BASE/batch_accumulator/output-rs0.jsonl | python3 -m json.tool"
echo ""

