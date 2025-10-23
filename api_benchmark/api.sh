#!/bin/bash

source ../.venv/bin/activate
python api_server.py \
    --host 0.0.0.0 \
    --port 16384 \
    --llm_path ../weights/rwkv7-g1a-0.4b-20250905-ctx4096 \
    --batch_size 128 \
    --max_tokens 2048
