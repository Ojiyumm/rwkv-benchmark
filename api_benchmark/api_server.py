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
import asyncio
from collections import deque
import torch


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

# ==================== 批量推理模型 ====================

class BatchCompletionRequest(BaseModel):
    prompts: List[str]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0

class BatchCompletionResponse(BaseModel):
    id: str
    object: str = "batch_completion"
    created: int
    results: List[str]
    batch_size: int
    inference_time: float
    speed_stats: Optional[Dict[str, Any]] = None


# ==================== Dynamic Batching ====================

class DynamicBatcher:
    """动态批处理器（可选功能）"""
    
    def __init__(self, engine, max_batch_size=128, max_wait_ms=50):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.stats = {'total_requests': 0, 'total_batches': 0, 'avg_batch_size': 0}
        self.enabled = False
    
    async def add_request(self, prompt: str, max_tokens: int) -> str:
        """添加请求到批处理队列"""
        request_id = uuid.uuid4().hex
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append({
                'id': request_id,
                'prompt': prompt,
                'max_tokens': max_tokens,
                'future': future
            })
            self.stats['total_requests'] += 1
            
            if len(self.pending_requests) >= self.max_batch_size:
                self.condition.notify()
        
        return await future
    
    async def batch_worker(self):
        """后台工作线程：处理批次"""
        print(f"[Dynamic Batching] Worker started (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)")
        
        while self.enabled:
            async with self.condition:
                if len(self.pending_requests) < self.max_batch_size:
                    try:
                        await asyncio.wait_for(
                            self.condition.wait(),
                            timeout=self.max_wait_ms / 1000
                        )
                    except asyncio.TimeoutError:
                        pass
                
                if not self.pending_requests:
                    continue
                
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
            
            await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[Dict]):
        """处理一个batch"""
        if not batch:
            return
        
        try:
            prompts = [req['prompt'] for req in batch]
            max_tokens = max(req['max_tokens'] for req in batch)
            
            tokens, _, _ = self.engine.generate_batch(prompts, max_length=max_tokens, noise=0.0)
            texts = self.engine.decode_tokens(tokens)
            
            for req, text in zip(batch, texts):
                if not req['future'].done():
                    req['future'].set_result(text)
            
            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = self.stats['total_requests'] / self.stats['total_batches']
            
        except Exception as e:
            print(f"[Dynamic Batching] Error: {e}")
            for req in batch:
                if not req['future'].done():
                    req['future'].set_exception(e)


# 全局引擎实例
engines: List[RWKVInferenceEngine] = []  # 多 GPU 支持
batcher: Optional[DynamicBatcher] = None
use_dynamic_batching: bool = False
gpu_ids_global: List[str] = []  # 存储 GPU IDs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化引擎
    global engines, batcher, use_dynamic_batching, gpu_ids_global
    
    # 从环境变量读取配置
    model_path = os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    use_dynamic_batching = os.getenv("USE_DYNAMIC_BATCHING", "false").lower() == "true"
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "128"))
    max_wait_ms = int(os.getenv("MAX_WAIT_MS", "50"))
    num_gpus = int(os.getenv("NUM_GPUS", "1"))
    gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")
    gpu_ids_global = gpu_ids  # 保存到全局变量
    
    print("\n" + "="*80)
    print("RWKV OpenAI-Compatible API Server")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Host: {api_host}")
    print(f"Port: {api_port}")
    print(f"GPUs: {num_gpus} ({gpu_ids})")
    print(f"Batch Size per GPU: {max_batch_size}")
    print(f"Total Batch Capacity: {max_batch_size * num_gpus}")
    print(f"Batch Mode: {'Dynamic Batching' if use_dynamic_batching else 'Batch Endpoint (Default)'}")
    if use_dynamic_batching:
        print(f"Max Wait: {max_wait_ms}ms")
    print("="*80 + "\n")
    
    print(f"Initializing {num_gpus} RWKV Inference Engine(s)...")
    
    # 多GPU模式：使用 torch.cuda.device 上下文管理器确保每个模型加载到正确的 GPU
    for i in range(num_gpus):
        gpu_id = int(gpu_ids[i]) if i < len(gpu_ids) else i
        print(f"  [GPU {gpu_id}] Loading model...")
        
        with torch.cuda.device(gpu_id):
            engine = RWKVInferenceEngine(
                model_path=model_path,
                seed=42 + i
            )
            engines.append(engine)
        print(f"  [GPU {gpu_id}] Ready (actual device: cuda:{gpu_id})!")
    
    print(f"All {num_gpus} engine(s) initialized successfully!")
    
    # 如果启用 Dynamic Batching，初始化 batcher（使用第一个 engine）
    batch_task = None
    if use_dynamic_batching:
        print("\nInitializing Dynamic Batcher...")
        batcher = DynamicBatcher(
            engine=engines[0],  # Dynamic batching 目前只用第一个 GPU
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms
        )
        batcher.enabled = True
        batch_task = asyncio.create_task(batcher.batch_worker())
        print("Dynamic Batcher started (using GPU 0)!")
    
    print(f"\nAPI Server ready at http://{api_host}:{api_port}")
    print("Press Ctrl+C to stop\n")
    yield
    
    # 关闭时清理
    print("\nShutting down...")
    if batch_task:
        if batcher:
            batcher.enabled = False
        batch_task.cancel()
        try:
            await batch_task
        except asyncio.CancelledError:
            pass
    print("Shutdown complete!")

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


@app.post("/v1/batch/completions", response_model=BatchCompletionResponse)
async def batch_completions(request: BatchCompletionRequest):
    """
    批量推理endpoint（最快！推荐用于评估）
    一次发送所有prompts，服务器批量处理后一起返回
    支持多 GPU 并行：自动将 batch 拆分到各个 GPU
    """
    if not engines:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        num_gpus = len(engines)
        total_prompts = len(request.prompts)
        
        print(f"[Batch Request] Received {total_prompts} prompts, max_tokens={request.max_tokens}, distributing to {num_gpus} GPUs")
        
        if num_gpus == 1:
            # 单 GPU，直接推理
            tokens, inference_time = engines[0].generate_batch(
                prompts=request.prompts,
                max_length=request.max_tokens,
                noise=0.0
            )
            texts = engines[0].decode_tokens(tokens)
            speed_stats = None
        else:
            # 多 GPU，拆分 batch 并行推理
            import concurrent.futures
            import numpy as np
            
            # 拆分 prompts 到各个 GPU
            chunk_size = (total_prompts + num_gpus - 1) // num_gpus
            prompt_chunks = [
                request.prompts[i:i+chunk_size] 
                for i in range(0, total_prompts, chunk_size)
            ]
            
            # 显示分配详情
            print(f"  Splitting {total_prompts} prompts across {num_gpus} GPUs:")
            for i, chunk in enumerate(prompt_chunks):
                gpu_id = int(gpu_ids_global[i]) if i < len(gpu_ids_global) else i
                print(f"    GPU {gpu_id}: {len(chunk)} prompts")
            
            start_time = time.time()
            
            # 并行调用各个 GPU
            def process_on_gpu(gpu_idx, prompts_chunk):
                if not prompts_chunk:
                    return [], 0
                # 确保在正确的 GPU 上执行
                actual_gpu_id = int(gpu_ids_global[gpu_idx]) if gpu_idx < len(gpu_ids_global) else gpu_idx
                
                chunk_size = len(prompts_chunk)
                print(f"  [GPU {actual_gpu_id}] Processing {chunk_size} prompts...")
                
                with torch.cuda.device(actual_gpu_id):
                    tokens, t = engines[gpu_idx].generate_batch(
                        prompts=prompts_chunk,
                        max_length=request.max_tokens,
                        noise=0.0
                    )
                    texts = engines[gpu_idx].decode_tokens(tokens)
                
                print(f"  [GPU {actual_gpu_id}] ✓ Completed {chunk_size} prompts in {t:.2f}s")
                return texts, t
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [
                    executor.submit(process_on_gpu, i, prompt_chunks[i])
                    for i in range(len(prompt_chunks))
                ]
                results = [f.result() for f in futures]
            
            inference_time = time.time() - start_time
            
            # 合并结果
            texts = []
            for chunk_texts, _ in results:
                texts.extend(chunk_texts)
            
            # 不返回详细的 speed stats（当前引擎不支持）
            speed_stats = None
            
            print(f"  ✓ All GPUs completed. Total time: {inference_time:.2f}s, Generated {len(texts)} responses")
        
        return BatchCompletionResponse(
            id=f"batch-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            results=texts,
            batch_size=len(request.prompts),
            inference_time=inference_time,
            speed_stats=speed_stats
        )
    
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {str(e)}"
        print(f"\n[ERROR] Batch completions failed: {error_details}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_details)


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI Completions API
    - 如果启用 Dynamic Batching：自动累积batch处理
    - 如果未启用：单个请求处理（性能较低）
    """
    if not engines:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Dynamic Batching 模式
    if use_dynamic_batching and batcher is not None:
        try:
            text = await batcher.add_request(request.prompt, request.max_tokens)
            prompt_tokens = len(engines[0].tokenizer.encode(request.prompt))
            
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[CompletionChoice(
                    index=0,
                    text=text,
                    finish_reason="length"
                )],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(text),
                    total_tokens=prompt_tokens + len(text)
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    try:
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        
        # 如果请求logprobs，使用专门的方法
        if request.logprobs is not None and request.logprobs > 0:
            results = engines[0].generate_with_logprobs(
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
            prompt_tokens = len(engines[0].tokenizer.encode(request.prompt))
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
            tokens, _, _ = engines[0].generate_batch(
                prompts=[request.prompt],
                max_length=request.max_tokens,
                noise=0.0
            )
            texts = engines[0].decode_tokens(tokens)
            generated_text = texts[0]
            generated_tokens = tokens[0].tolist()
            
            prompt_tokens = len(engines[0].tokenizer.encode(request.prompt))
            
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
    if not engines:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = {
        "status": "running",
        "model": os.getenv("MODEL_PATH", "unknown"),
        "num_gpus": len(engines),
        "batch_mode": "dynamic_batching" if use_dynamic_batching else "batch_endpoint"
    }
    
    if use_dynamic_batching and batcher is not None:
        stats["dynamic_batching"] = {
            "enabled": True,
            "stats": batcher.stats,
            "pending_requests": len(batcher.pending_requests)
        }
    
    return stats


@app.get("/health")
async def health_check():
    """健康检查"""
    if not engines:
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
    parser.add_argument("--batch_mode", type=str, default="endpoint", 
                        choices=["endpoint", "dynamic"],
                        help="Batch mode: 'endpoint' (default, fastest) or 'dynamic' (for concurrent requests)")
    parser.add_argument("--max_wait_ms", type=int, default=50, 
                        help="Max wait time for dynamic batching (ms)")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.llm_path
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["MAX_BATCH_SIZE"] = str(args.batch_size)
    os.environ["MAX_TOKENS"] = str(args.max_tokens)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["USE_DYNAMIC_BATCHING"] = "true" if args.batch_mode == "dynamic" else "false"
    os.environ["MAX_WAIT_MS"] = str(args.max_wait_ms)
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info"
    )

