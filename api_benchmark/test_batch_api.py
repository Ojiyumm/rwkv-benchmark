#!/usr/bin/env python3
"""
测试两种 Batch 模式的性能
"""

import requests
import time
import asyncio
import aiohttp


API_URL = "http://localhost:8001"


def test_batch_endpoint(prompts, max_tokens=20):
    """测试 Batch Endpoint（方案1，最快）"""
    print(f"\n{'='*60}")
    print("测试 Batch Endpoint")
    print(f"{'='*60}")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/v1/batch/completions",
        json={
            "prompts": prompts,
            "max_tokens": max_tokens
        },
        timeout=300
    )
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 成功")
        print(f"   Batch Size: {result['batch_size']}")
        print(f"   推理时间: {result['inference_time']:.3f}s")
        print(f"   总时间: {elapsed:.3f}s")
        if result.get('speed_stats'):
            stats = result['speed_stats']
            print(f"   速度: forward {stats['forward_tps']:.2f} tok/s, full {stats['full_tps']:.2f} tok/s")
        print(f"   结果示例: {result['results'][0][:50]}...")
        return result['results'], elapsed
    else:
        print(f"❌ 失败: {response.text}")
        return None, elapsed


async def test_dynamic_batching(prompts, max_tokens=20):
    """测试 Dynamic Batching（方案2）"""
    print(f"\n{'='*60}")
    print("测试 Dynamic Batching")
    print(f"{'='*60}")
    
    async def single_request(session, prompt):
        async with session.post(
            f"{API_URL}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            result = await response.json()
            return result["choices"][0]["text"]
    
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [single_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"✅ 成功")
    print(f"   Batch Size: {len(prompts)}")
    print(f"   总时间: {elapsed:.3f}s")
    print(f"   结果示例: {results[0][:50]}...")
    
    return results, elapsed


def main():
    # 测试数据
    prompts = [
        "The capital of France is",
        "1+1=",
        "Today is",
        "Hello world",
    ] * 32  # 128个请求
    
    print(f"\n{'='*80}")
    print(f"API Batch 性能测试")
    print(f"{'='*80}")
    print(f"API URL: {API_URL}")
    print(f"测试请求数: {len(prompts)}")
    print(f"{'='*80}")
    
    # 检查服务器状态
    try:
        response = requests.get(f"{API_URL}/stats")
        stats = response.json()
        print(f"\n服务器状态:")
        print(f"  模式: {stats['batch_mode']}")
        print(f"  状态: {stats['status']}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return
    
    # 测试1: Batch Endpoint
    results1, time1 = test_batch_endpoint(prompts, max_tokens=20)
    
    # 测试2: Dynamic Batching (如果启用)
    if stats['batch_mode'] == 'dynamic_batching':
        results2, time2 = asyncio.run(test_dynamic_batching(prompts, max_tokens=20))
        
        print(f"\n{'='*60}")
        print("性能对比")
        print(f"{'='*60}")
        print(f"Batch Endpoint:    {time1:.3f}s  ← 更快！")
        print(f"Dynamic Batching:  {time2:.3f}s")
        print(f"差距:              {time2-time1:.3f}s ({(time2/time1-1)*100:.1f}%)")
    else:
        print(f"\n💡 提示: 当前服务器使用 Batch Endpoint 模式")
        print(f"   要测试 Dynamic Batching，重启服务器时加上: --batch_mode dynamic")
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

