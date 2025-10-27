#!/bin/bash


export HF_HOME="~/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com



export OPENAI_API_KEY="EMPTY"

lm_eval --model openai-completions \
    --model_args base_url="http://192.168.0.82:8001/v1/completions",model=davinci-002,tokenized_requests=false \
    --tasks aime24 \
    --output_path ./eval_results 
   
