#!/bin/bash

# RWKV API Server 启动脚本
# 
# Batch Mode:
#   endpoint (default) - 使用 /v1/batch/completions endpoint，最快！推荐用于评估
#   dynamic            - 使用 Dynamic Batching，自动累积请求，适合在线服务
#
# 多 GPU 支持:
#   设置 NUM_GPUS 和 CUDA_VISIBLE_DEVICES 环境变量
#   例如 4 张卡: export NUM_GPUS=4 && export CUDA_VISIBLE_DEVICES=0,1,2,3

# 单卡模式（默认）
# export NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0

# 多卡模式（4 张卡）
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

python api_server.py \
    --host 192.168.0.157 \
    --port 8001 \
    --llm_path /home/rwkv/model/rwkv/rwkv7-g0a2-7.2b-20251005-ctx4096 \
    --batch_size 320 \
    --max_tokens 4096 \
    --batch_mode endpoint \
    --max_wait_ms 50

# 使用 Dynamic Batching 模式：
# --batch_mode dynamic --max_wait_ms 50
