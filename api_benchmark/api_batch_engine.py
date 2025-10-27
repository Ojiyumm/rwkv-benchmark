"""
API Client Adapter
接口和 batch_engine.py 完全一致，可以直接替换到评估框架中
"""

import requests
import asyncio
import aiohttp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time


class APIInferenceEngine:
    """
    API推理引擎（接口和 RWKVInferenceEngine 一致）
    通过 HTTP API 调用 Dynamic Batching 服务器
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: int = 300  # 5分钟超时（等待batch处理）
    ):
        """
        初始化API引擎
        
        Args:
            api_url: API服务器地址
            timeout: 请求超时时间（秒）
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        
        print(f'\nConnecting to API server at {self.api_url}...')
        
        # 检查服务器健康状态
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.json()["status"] != "healthy":
                raise RuntimeError("API server is not healthy")
            print(f'API server connected successfully!\n')
        except Exception as e:
            raise RuntimeError(f"Failed to connect to API server: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 100,
        noise: float = 0.0,
        return_speed: bool = False
    ) -> Tuple[np.ndarray, float, Optional[Dict]]:
        """
        批量生成文本（通过异步并发请求，服务器端会自动batch）
        
        Args:
            prompts: 提示列表
            max_length: 最大生成长度
            noise: 采样噪声（API暂不支持）
            return_speed: 是否返回速度统计
            
        Returns:
            tokens: numpy数组（这里返回的是文本，为了兼容性）
            inference_time: 推理总时间
            speed_stats: 速度统计（如果 return_speed=True）
        """
        start_time = time.time()
        
        # 使用异步并发发送所有请求
        # 服务器端会自动累积并batch处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self._async_batch_generate(prompts, max_length)
            )
        finally:
            loop.close()
        
        inference_time = time.time() - start_time
        
        # 为了兼容性，将文本转换为伪tokens（实际是字符串）
        # 真实场景应该返回实际的token ids
        tokens = np.array(results, dtype=object)
        
        # 速度统计
        speed_stats = None
        if return_speed:
            # 从API获取服务器端统计
            try:
                stats_response = requests.get(f"{self.api_url}/stats", timeout=5)
                server_stats = stats_response.json()
                
                # 估算速度（基于总时间）
                total_chars = sum(len(r) for r in results)
                speed_stats = {
                    'batch_size': len(prompts),
                    'forward_tps': total_chars / inference_time,  # 粗略估计
                    'full_tps': total_chars / inference_time,
                    'step_forward_ms_p50': inference_time * 1000 / max_length,
                    'step_full_ms_p50': inference_time * 1000 / max_length,
                    'server_stats': server_stats['stats']
                }
            except:
                pass
        
        return tokens, inference_time, speed_stats
    
    async def _async_batch_generate(
        self,
        prompts: List[str],
        max_length: int
    ) -> List[str]:
        """异步并发发送所有请求"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._async_single_request(session, prompt, max_length)
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            return results
    
    async def _async_single_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_length: int
    ) -> str:
        """单个异步请求"""
        url = f"{self.api_url}/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_length,
            "temperature": 1.0
        }
        
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as response:
            result = await response.json()
            return result["text"]
    
    def decode_tokens(
        self,
        tokens: np.ndarray,
        utf8_errors: str = "ignore"
    ) -> List[str]:
        """
        解码tokens为文本
        （由于API返回的已经是文本，直接返回）
        
        Args:
            tokens: "tokens"数组（实际是文本）
            utf8_errors: UTF-8错误处理方式
            
        Returns:
            解码后的文本列表
        """
        # API返回的已经是文本，直接转换
        return [str(t) for t in tokens]
    
    def save_qa_to_jsonl(
        self,
        prompts: List[str],
        responses: List[str],
        task_name: str,
        output_dir: str = ".",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """保存问答对到jsonl文件（和本地引擎接口一致）"""
        import json
        import os
        from datetime import datetime
        
        if len(prompts) != len(responses):
            raise ValueError(f"Prompts and responses must have the same length")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                record = {
                    "index": i,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    record.update(metadata)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(prompts)} Q&A pairs to {filepath}")
        return filepath


if __name__ == "__main__":
    # 测试
    print("Testing API Inference Engine...")
    
    engine = APIInferenceEngine(api_url="http://localhost:8000")
    
    # 测试batch生成
    prompts = ["Hello", "The cat", "今天天气"]
    print(f"\nTesting batch generation with {len(prompts)} prompts...")
    
    tokens, inference_time, speed_stats = engine.generate_batch(
        prompts,
        max_length=20,
        return_speed=True
    )
    texts = engine.decode_tokens(tokens)
    
    print(f"\nResults:")
    print(f"  Inference time: {inference_time:.4f}s")
    if speed_stats:
        print(f"  Server stats: {speed_stats.get('server_stats')}")
    
    for prompt, text in zip(prompts, texts):
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {text}")
    
    print("\nTest completed!")

