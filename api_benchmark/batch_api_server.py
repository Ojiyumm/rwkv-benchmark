"""
Dynamic Batching API Server
支持累积多个请求，达到batch_size后一次性推理并返回所有结果
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager
import os

from batch_engine import RWKVInferenceEngine


# API 数据模型
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0

class CompletionResponse(BaseModel):
    id: str
    text: str
    prompt: str
    usage: Dict[str, int]


class DynamicBatcher:
    """动态批处理器"""
    
    def __init__(self, engine, max_batch_size=128, max_wait_ms=50):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.pending_requests = []
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0
        }
    
    async def add_request(self, prompt: str, max_tokens: int) -> str:
        """
        添加请求到批处理队列
        返回生成的文本
        """
        request_id = uuid.uuid4().hex
        future = asyncio.Future()
        
        async with self.lock:
            # 添加到队列
            self.pending_requests.append({
                'id': request_id,
                'prompt': prompt,
                'max_tokens': max_tokens,
                'future': future
            })
            
            self.stats['total_requests'] += 1
            
            # 如果达到batch大小，触发处理
            if len(self.pending_requests) >= self.max_batch_size:
                self.condition.notify()
        
        # 等待结果
        result = await future
        return result
    
    async def batch_worker(self):
        """后台工作线程：处理批次"""
        print(f"Batch worker started (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)")
        
        while True:
            async with self.condition:
                # 等待条件：有请求 且 (达到batch大小 或 超时)
                if len(self.pending_requests) < self.max_batch_size:
                    try:
                        await asyncio.wait_for(
                            self.condition.wait(),
                            timeout=self.max_wait_ms / 1000
                        )
                    except asyncio.TimeoutError:
                        pass
                
                # 如果还是没有请求，继续等待
                if not self.pending_requests:
                    continue
                
                # 取出一个batch
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
            
            # 批量推理（不持有锁）
            await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[Dict]):
        """处理一个batch"""
        batch_size = len(batch)
        print(f"\n[Batch Worker] Processing batch of {batch_size} requests...")
        
        try:
            # 提取所有prompts
            prompts = [req['prompt'] for req in batch]
            max_tokens = max(req['max_tokens'] for req in batch)
            
            # 批量推理
            start_time = time.time()
            tokens, inference_time, speed_stats = self.engine.generate_batch(
                prompts=prompts,
                max_length=max_tokens,
                noise=0.0,
                return_speed=True
            )
            texts = self.engine.decode_tokens(tokens)
            total_time = time.time() - start_time
            
            # 显示速度统计
            if speed_stats:
                print(f"[Batch Worker] Completed in {total_time:.3f}s")
                print(f"[Batch Worker] Speed: forward {speed_stats['forward_tps']:.2f} tok/s, full {speed_stats['full_tps']:.2f} tok/s")
            
            # 分发结果给各个请求
            for req, text in zip(batch, texts):
                if not req['future'].done():
                    req['future'].set_result(text)
            
            # 更新统计
            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = (
                self.stats['total_requests'] / self.stats['total_batches']
            )
            
        except Exception as e:
            print(f"[Batch Worker] Error: {e}")
            # 出错时通知所有请求
            for req in batch:
                if not req['future'].done():
                    req['future'].set_exception(e)


# 全局变量
engine: Optional[RWKVInferenceEngine] = None
batcher: Optional[DynamicBatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动和关闭时的处理"""
    global engine, batcher
    
    # 启动时初始化
    model_path = os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096")
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "128"))
    max_wait_ms = int(os.getenv("MAX_WAIT_MS", "50"))
    
    print("\n" + "="*80)
    print("RWKV Dynamic Batching API Server")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Max Batch Size: {max_batch_size}")
    print(f"Max Wait: {max_wait_ms}ms")
    print("="*80 + "\n")
    
    print("Initializing RWKV Inference Engine...")
    engine = RWKVInferenceEngine(model_path=model_path, seed=42)
    print("Engine initialized!\n")
    
    print("Starting Dynamic Batcher...")
    batcher = DynamicBatcher(
        engine=engine,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms
    )
    
    # 启动后台工作线程
    batch_task = asyncio.create_task(batcher.batch_worker())
    
    print("API Server ready!\n")
    
    yield
    
    # 关闭时清理
    print("\nShutting down...")
    batch_task.cancel()
    print("Shutdown complete!")


# 创建FastAPI应用
app = FastAPI(
    title="RWKV Dynamic Batching API",
    description="Automatic batching for high throughput",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    单个请求endpoint
    服务器会自动累积到batch_size后批量处理
    """
    if batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")
    
    try:
        # 提交请求（会自动batch）
        text = await batcher.add_request(
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        
        # 返回结果
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            text=text,
            prompt=request.prompt,
            usage={
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(text),
                "total_tokens": len(request.prompt) + len(text)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    if batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")
    
    return {
        "status": "running",
        "stats": batcher.stats,
        "pending_requests": len(batcher.pending_requests)
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if engine is None or batcher is None:
        return {"status": "unhealthy"}
    return {"status": "healthy"}


@app.get("/")
async def root():
    """API信息"""
    return {
        "name": "RWKV Dynamic Batching API",
        "version": "1.0.0",
        "endpoints": {
            "completions": "/v1/completions",
            "stats": "/stats",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RWKV Dynamic Batching API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model-path", type=str, required=True, help="Path to RWKV model")
    parser.add_argument("--batch-size", type=int, default=128, help="Max batch size")
    parser.add_argument("--max-wait-ms", type=int, default=50, help="Max wait time (ms)")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MAX_BATCH_SIZE"] = str(args.batch_size)
    os.environ["MAX_WAIT_MS"] = str(args.max_wait_ms)
    
    uvicorn.run(
        "batch_api_server:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info"
    )

