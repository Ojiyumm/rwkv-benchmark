#!/bin/bash
python api_server.py \
    --host 192.168.0.82 \
    --port 8001 \
    --llm_path /home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096 \
    --batch_size 128 \
    --max_tokens 2048
