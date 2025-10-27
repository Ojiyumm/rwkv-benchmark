#!/usr/bin/env python3
"""
快速测试 nemo-skills + RWKV 集成
"""

import asyncio
import json
import time
from pathlib import Path

# 测试数据
TEST_DATA = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
] * 10  # 30 samples


async def test_rwkv_batch_model():
    """测试 RWKVBatchModel"""
    print("\n" + "="*60)
    print("测试 1: RWKVBatchModel (Batch Accumulator)")
    print("="*60)
    
    try:
        from nemo_skills.inference.model.rwkv_batch import RWKVBatchModel
        
        model = RWKVBatchModel(
            host="192.168.0.82",
            port="8001",
            model="rwkv-7-world",
            batch_size=16,
            max_wait_ms=100,
        )
        
        print(f"✅ Model initialized: {model.base_url}")
        
        # 并发测试
        start_time = time.time()
        
        tasks = []
        for i, item in enumerate(TEST_DATA):
            prompt = f"Question: {item['question']}\nAnswer:"
            task = model.generate_async(
                prompt=prompt,
                tokens_to_generate=50,
                temperature=0.0,
            )
            tasks.append((i, task))
        
        print(f"发送 {len(tasks)} 个并发请求...")
        
        results = []
        for i, task in tasks:
            result = await task
            results.append(result)
            if i % 10 == 0:
                print(f"  完成 {i+1}/{len(tasks)}")
        
        elapsed = time.time() - start_time
        
        print(f"\n结果:")
        print(f"  总请求数: {len(TEST_DATA)}")
        print(f"  总时间: {elapsed:.2f}s")
        print(f"  平均延迟: {elapsed/len(TEST_DATA)*1000:.2f}ms/request")
        print(f"  吞吐量: {len(TEST_DATA)/elapsed:.2f} requests/s")
        
        print(f"\n示例输出:")
        for i in range(min(3, len(results))):
            print(f"  [{i}] {results[i]['generation'][:100]}...")
        
        await model.accumulator.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rwkv_direct_model():
    """测试 RWKVDirectModel"""
    print("\n" + "="*60)
    print("测试 2: RWKVDirectModel (Direct API)")
    print("="*60)
    
    try:
        from nemo_skills.inference.model.rwkv_batch import RWKVDirectModel
        
        model = RWKVDirectModel(
            host="192.168.0.82",
            port="8001",
            model="rwkv-7-world",
        )
        
        print(f"✅ Model initialized: {model.base_url}")
        
        # 测试少量请求（direct 模式较慢）
        test_samples = TEST_DATA[:5]
        
        start_time = time.time()
        
        results = []
        for i, item in enumerate(test_samples):
            prompt = f"Question: {item['question']}\nAnswer:"
            result = await model.generate_async(
                prompt=prompt,
                tokens_to_generate=50,
                temperature=0.0,
            )
            results.append(result)
            print(f"  完成 {i+1}/{len(test_samples)}")
        
        elapsed = time.time() - start_time
        
        print(f"\n结果:")
        print(f"  总请求数: {len(test_samples)}")
        print(f"  总时间: {elapsed:.2f}s")
        print(f"  平均延迟: {elapsed/len(test_samples)*1000:.2f}ms/request")
        
        print(f"\n示例输出:")
        for i in range(len(results)):
            print(f"  [{i}] {results[i]['generation'][:100]}...")
        
        await model.client.aclose()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_server():
    """测试 API 服务器是否运行"""
    print("\n" + "="*60)
    print("测试 0: API 服务器连接")
    print("="*60)
    
    import requests
    
    try:
        response = requests.get("http://192.168.0.82:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 服务器运行正常")
            
            stats_response = requests.get("http://192.168.0.82:8001/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"   模式: {stats.get('batch_mode', 'unknown')}")
                print(f"   状态: {stats.get('status', 'unknown')}")
            
            return True
        else:
            print(f"❌ API 服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到 API 服务器: {e}")
        print(f"\n请先启动服务器:")
        print(f"  cd /home/rwkv/Peter/rwkveval/api_benchmark")
        print(f"  bash api.sh")
        return False


async def main():
    print("\n" + "="*80)
    print("Nemo-Skills + RWKV 集成测试")
    print("="*80)
    
    # 测试 API 服务器
    if not test_api_server():
        return
    
    # 测试 Batch Model
    batch_ok = await test_rwkv_batch_model()
    
    # 测试 Direct Model
    direct_ok = await test_rwkv_direct_model()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"  Batch Accumulator: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    print(f"  Direct API:        {'✅ PASS' if direct_ok else '❌ FAIL'}")
    
    if batch_ok and direct_ok:
        print("\n🎉 所有测试通过！可以开始使用 nemo-skills 评估了。")
        print("\n查看使用文档:")
        print("  cat /home/rwkv/Peter/rwkveval/api_benchmark/NEMO_SKILLS_INTEGRATION.md")
    else:
        print("\n⚠️  部分测试失败，请检查配置")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

