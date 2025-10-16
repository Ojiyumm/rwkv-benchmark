"""
FastAPI Server with OpenAI-Compatible API
使用BatchInferenceEngine提供标准OpenAI API接口
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager

from batch_engine import RWKVInferenceEngine
import os


# OpenAI API兼容的数据模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    logprobs: Optional[int] = None  # 返回top-N logprobs
    echo: Optional[bool] = False  # 是否回显prompt

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class Logprobs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str
    logprobs: Optional[Logprobs] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# 全局引擎实例
engine: Optional[RWKVInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化引擎
    global engine
    
    # 从环境变量读取配置
    model_path = os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    
    print("\n" + "="*80)
    print("RWKV OpenAI-Compatible API Server")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Host: {api_host}")
    print(f"Port: {api_port}")
    print("="*80 + "\n")
    
    print("Initializing RWKV Inference Engine...")
    engine = RWKVInferenceEngine(
        model_path=model_path,
        seed=42
    )
    print("Engine initialized successfully!")
    print(f"\nAPI Server ready at http://{api_host}:{api_port}")
    print("Press Ctrl+C to stop\n")
    yield
    # 关闭时清理
    print("\nShutting down engine...")
    print("Engine shutdown complete!")

# 创建FastAPI应用
app = FastAPI(
    title="RWKV OpenAI-Compatible API",
    description="High-performance RWKV inference with OpenAI-compatible endpoints",
    version="1.0.0",
    lifespan=lifespan
)


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """将chat messages格式化为prompt"""
    # 简单实现：直接拼接
    # 可以根据模型需要自定义格式
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}\n")
    
    # 添加最后的Assistant提示
    prompt_parts.append("Assistant:")
    return "".join(prompt_parts)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI Chat Completions API"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # 格式化prompt
        prompt = format_chat_prompt(request.messages)
        
        # 生成
        tokens, _ = engine.generate_batch(
            prompts=[prompt],
            max_length=request.max_tokens,
            noise=0.0  # 使用deterministic采样
        )
        texts = engine.decode_tokens(tokens)
        generated_text = texts[0]
        generated_tokens = tokens[0].tolist()
        
        # 构造OpenAI格式响应
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason="length" if len(generated_tokens) >= request.max_tokens else "stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt),
                completion_tokens=len(generated_tokens),
                total_tokens=len(prompt) + len(generated_tokens)
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI Completions API"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        
        # 如果请求logprobs，使用专门的方法
        if request.logprobs is not None and request.logprobs > 0:
            results = engine.generate_with_logprobs(
                prompts=[request.prompt],
                max_length=request.max_tokens,
                echo=request.echo,
                top_logprobs=request.logprobs
            )
            result = results[0]
            
            # 构造logprobs响应
            logprobs_obj = Logprobs(
                tokens=result['token_strs'],
                token_logprobs=result['logprobs'],
                top_logprobs=result['top_logprobs'],
                text_offset=[0] * len(result['token_strs'])
            )
            
            # 计算prompt tokens数量
            prompt_tokens = len(engine.tokenizer.encode(request.prompt))
            completion_tokens = len(result['tokens']) - (prompt_tokens if request.echo else 0)
            
            return CompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result['text'],
                        finish_reason="length",
                        logprobs=logprobs_obj
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=len(result['tokens'])
                )
            )
        else:
            # 普通生成（不需要logprobs）
            tokens, _ = engine.generate_batch(
                prompts=[request.prompt],
                max_length=request.max_tokens,
                noise=0.0
            )
            texts = engine.decode_tokens(tokens)
            generated_text = texts[0]
            generated_tokens = tokens[0].tolist()
            
            prompt_tokens = len(engine.tokenizer.encode(request.prompt))
            
            return CompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=generated_text,
                        finish_reason="length" if len(generated_tokens) >= request.max_tokens else "stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(generated_tokens),
                    total_tokens=prompt_tokens + len(generated_tokens)
                )
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "davinci-002",  # lm_eval需要这个用于loglikelihood
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rwkv"
            },
            {
                "id": "gpt-3.5-turbo",  # 伪装成gpt-3.5-turbo绕过tiktoken检查
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rwkv"
            },
            {
                "id": "rwkv-7-world",  # 真实名称
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rwkv"
            }
        ]
    }


@app.get("/stats")
async def get_stats():
    """获取引擎性能统计（额外端点）"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "status": "running",
        "model": os.getenv("MODEL_PATH", "unknown")
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Engine not initialized"}
        )
    return {"status": "healthy"}


@app.get("/")
async def root():
    """API信息"""
    return {
        "name": "RWKV OpenAI-Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "stats": "/stats",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RWKV API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to RWKV model")
    parser.add_argument("--batch_size", type=int, default=128, help="Max batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.llm_path
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["MAX_BATCH_SIZE"] = str(args.batch_size)
    os.environ["MAX_TOKENS"] = str(args.max_tokens)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info"
    )

