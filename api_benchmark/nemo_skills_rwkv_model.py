"""
RWKV Model for nemo-skills with Batch Inference Support
使 nemo-skills 支持 RWKV 批量推理的适配器
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

from nemo_skills.inference.model.base import BaseModel, EndpointType
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class BatchAccumulator:
    """
    累积并发请求，然后批量调用 /v1/batch/completions
    """
    
    def __init__(self, api_url: str, batch_size: int = 128, max_wait_ms: int = 50):
        self.api_url = api_url
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        
        self.pending_requests: List[Dict] = []
        self.lock = asyncio.Lock()
        self.last_flush_time = time.time()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        
        # Background task to flush periodically
        self.flush_task = None
        self.running = True
        
        LOG.info(f"BatchAccumulator initialized: batch_size={batch_size}, max_wait={max_wait_ms}ms")
    
    async def start_background_flusher(self):
        """启动后台flush任务"""
        self.flush_task = asyncio.create_task(self._periodic_flush())
    
    async def _periodic_flush(self):
        """定期检查并flush batch"""
        while self.running:
            await asyncio.sleep(self.max_wait_ms)
            
            async with self.lock:
                elapsed = time.time() - self.last_flush_time
                
                # 如果有pending请求且超过等待时间，flush
                if self.pending_requests and elapsed >= self.max_wait_ms:
                    await self._flush_batch_unsafe()
    
    async def add_request(self, prompt: str, max_tokens: int, **kwargs) -> str:
        """添加一个请求到batch"""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'future': future,
                'kwargs': kwargs
            })
            
            # 如果batch满了，立即flush
            if len(self.pending_requests) >= self.batch_size:
                await self._flush_batch_unsafe()
        
        # 等待结果
        return await future
    
    async def _flush_batch_unsafe(self):
        """Flush当前batch（调用者需持有lock）"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests
        self.pending_requests = []
        self.last_flush_time = time.time()
        
        # 释放锁后执行API调用（避免阻塞其他请求）
        asyncio.create_task(self._process_batch(batch))
    
    async def _process_batch(self, batch: List[Dict]):
        """处理一个batch的请求"""
        try:
            prompts = [req['prompt'] for req in batch]
            max_tokens = max(req['max_tokens'] for req in batch)
            
            LOG.info(f"Processing batch of {len(batch)} requests, max_tokens={max_tokens}")
            
            start_time = time.time()
            
            # 调用 /v1/batch/completions
            response = await self.client.post(
                f"{self.api_url}/v1/batch/completions",
                json={
                    'prompts': prompts,
                    'max_tokens': max_tokens
                }
            )
            response.raise_for_status()
            
            result = response.json()
            texts = result['results']
            
            elapsed = time.time() - start_time
            
            # 记录性能
            if 'speed_stats' in result:
                stats = result['speed_stats']
                LOG.info(
                    f"Batch processed in {elapsed:.2f}s: "
                    f"forward {stats['forward_tps']:.2f} tok/s, "
                    f"full {stats['full_tps']:.2f} tok/s"
                )
            
            # 分发结果
            for req, text in zip(batch, texts):
                if not req['future'].done():
                    req['future'].set_result(text)
        
        except Exception as e:
            LOG.error(f"Error processing batch: {e}")
            # 失败时分发异常
            for req in batch:
                if not req['future'].done():
                    req['future'].set_exception(e)
    
    async def flush_all(self):
        """强制flush所有pending请求"""
        async with self.lock:
            if self.pending_requests:
                await self._flush_batch_unsafe()
    
    async def stop(self):
        """停止并清理"""
        self.running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        await self.flush_all()
        await self.client.aclose()


class RWKVBatchModel(BaseModel):
    """
    RWKV 批量推理模型
    自动累积并发请求，调用 /v1/batch/completions 进行批量推理
    """
    
    MODEL_PROVIDER = "openai"  # 兼容性
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "8001",
        model: str = "rwkv-7-world",
        base_url: str | None = None,
        batch_size: int = 128,
        max_wait_ms: int = 50,
        **kwargs,
    ):
        # 不调用 super().__init__，因为我们不使用 litellm
        self.model_name_or_path = model
        self.server_host = host
        self.server_port = port
        
        if base_url is None:
            base_url = f"http://{host}:{port}"
        
        self.base_url = base_url
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        
        # 创建 batch accumulator
        self.accumulator = BatchAccumulator(
            api_url=base_url,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms
        )
        
        # 启动后台flusher
        asyncio.create_task(self.accumulator.start_background_flusher())
        
        LOG.info(f"RWKVBatchModel initialized: {base_url}, batch_size={batch_size}")
    
    def _build_chat_request_params(self, **kwargs) -> dict:
        """Dummy implementation for compatibility"""
        return kwargs
    
    def _build_completion_request_params(self, **kwargs) -> dict:
        """Dummy implementation for compatibility"""
        return kwargs
    
    async def generate_async(
        self,
        prompt: str | list[dict],
        endpoint_type: EndpointType = None,
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
    ) -> dict:
        """
        重写 generate_async，使用批量推理
        """
        # 转换 prompt 为字符串
        if isinstance(prompt, list):
            # OpenAI chat format to string
            prompt_str = "\n".join([msg.get('content', '') for msg in prompt if msg.get('content')])
        else:
            prompt_str = prompt
        
        # 添加请求到 accumulator
        result_text = await self.accumulator.add_request(
            prompt=prompt_str,
            max_tokens=tokens_to_generate or 100,
            temperature=temperature,
            top_p=top_p,
            random_seed=random_seed,
        )
        
        # 返回 nemo-skills 期望的格式
        return {
            "generation": result_text,
            "num_generated_tokens": len(result_text.split()),  # Approximate
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.accumulator.stop()


class RWKVDirectModel(BaseModel):
    """
    RWKV 直接调用模型（不使用 litellm）
    适用于单个请求场景或已经由服务器端做 Dynamic Batching 的情况
    """
    
    MODEL_PROVIDER = "openai"
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "8001",
        model: str = "rwkv-7-world",
        base_url: str | None = None,
        **kwargs,
    ):
        self.model_name_or_path = model
        self.server_host = host
        self.server_port = port
        
        if base_url is None:
            base_url = f"http://{host}:{port}"
        
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        
        LOG.info(f"RWKVDirectModel initialized: {base_url}")
    
    def _build_chat_request_params(self, **kwargs) -> dict:
        return kwargs
    
    def _build_completion_request_params(self, **kwargs) -> dict:
        return kwargs
    
    async def generate_async(
        self,
        prompt: str | list[dict],
        endpoint_type: EndpointType = None,
        tokens_to_generate: int | None = None,
        **kwargs
    ) -> dict:
        """直接调用 /v1/completions"""
        
        # 转换 prompt
        if isinstance(prompt, list):
            prompt_str = "\n".join([msg.get('content', '') for msg in prompt if msg.get('content')])
        else:
            prompt_str = prompt
        
        # 调用 API
        response = await self.client.post(
            f"{self.base_url}/v1/completions",
            json={
                'prompt': prompt_str,
                'max_tokens': tokens_to_generate or 100,
            }
        )
        response.raise_for_status()
        
        result = response.json()
        text = result['choices'][0]['text']
        
        return {
            "generation": text,
            "num_generated_tokens": len(text.split()),
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

