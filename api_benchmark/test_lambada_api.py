"""
LAMBADA Evaluation using API
通过API接口测试LAMBADA数据集，使用并发请求加速
"""

import json
import requests
import numpy as np
from tqdm import tqdm
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def test_lambada_via_api(
    api_url: str = "http://0.0.0.0:16384",
    lambada_path: str = "/public/home/ssjxzkz/Projects/rwkv-benchmark/eval/lambada_test.jsonl",
    max_samples: int = None,
    max_workers: int = 32
):
    """
    通过API测试LAMBADA数据集，使用并发请求加速
    
    Args:
        api_url: API服务器地址
        lambada_path: LAMBADA数据集路径
        max_samples: 最大测试样本数（None表示全部）
        max_workers: 并发线程数
    """
    print(f"\n{'='*80}")
    print(f"LAMBADA Evaluation via API (Concurrent)")
    print(f"{'='*80}")
    print(f"API URL: {api_url}")
    print(f"Dataset: {lambada_path}")
    print(f"Max workers: {max_workers}")
    print()
    
    # 检查API健康状态
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        if health_resp.status_code != 200:
            print(f"Error: API server not healthy")
            return
        print("✓ API server is healthy")
    except Exception as e:
        print(f"Error: Cannot connect to API server at {api_url}")
        print(f"  {e}")
        return
    
    # 加载LAMBADA数据集
    try:
        with open(lambada_path, "r", encoding="utf-8") as f:
            todo = [json.loads(line) for line in f]
            # Split into context and target word
            todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]
    except FileNotFoundError:
        print(f"Error: LAMBADA dataset not found at {lambada_path}")
        return
    
    if max_samples:
        todo = todo[:max_samples]
    
    print(f"Loaded {len(todo)} examples")
    print("Starting concurrent evaluation...\n")
    
    # 统计变量（线程安全）
    lock = threading.Lock()
    results = []
    
    def process_sample(idx_context_target):
        """处理单个样本"""
        idx, context, target = idx_context_target
        
        try:
            resp = requests.post(
                f"{api_url}/v1/completions",
                json={
                    "model": "rwkv-7-world",
                    "prompt": context,
                    "max_tokens": 10,
                    "temperature": 1.0
                },
                timeout=30
            )
            
            if resp.status_code != 200:
                return (idx, False, f"API error: {resp.status_code}")
            
            result = resp.json()
            generated_text = result['choices'][0]['text']
            
            # 简单判断：检查生成的文本开头是否包含目标词
            target_word = target.strip()
            generated_word = generated_text.strip().split()[0] if generated_text.strip() else ""
            
            is_correct = (generated_word.lower() == target_word.lower())
            
            return (idx, is_correct, None)
        
        except Exception as e:
            return (idx, False, str(e))
    
    # 准备任务
    tasks = [(i, context, target) for i, (context, target) in enumerate(todo)]
    
    # 并发执行
    start_time = time.time()
    correct_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_sample, task) for task in tasks]
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                idx, is_correct, error = future.result()
                
                if error:
                    error_count += 1
                elif is_correct:
                    correct_count += 1
                
                results.append((idx, is_correct, error))
            except Exception as e:
                error_count += 1
    
    # 最终结果
    elapsed = time.time() - start_time
    total_count = len(results)
    successful_count = total_count - error_count
    accuracy = correct_count / successful_count * 100 if successful_count > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Final Results:")
    print(f"  Total samples: {total_count}")
    print(f"  Successful: {successful_count}")
    print(f"  Errors: {error_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_count/elapsed:.2f} samples/s")
    print(f"{'='*80}\n")
    
    # 保存结果
    output_file = f"lambada_api_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_samples': total_count,
            'successful': successful_count,
            'errors': error_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'time': elapsed,
            'speed': total_count/elapsed,
            'api_url': api_url,
            'max_workers': max_workers
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")


def quick_test(api_url: str = "http://localhost:8000"):
    """快速测试API功能"""
    print(f"\n{'='*80}")
    print(f"Quick API Test")
    print(f"{'='*80}")
    print(f"API URL: {api_url}\n")
    
    # 测试1: Health check
    print("[1] Health Check...")
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")
    except Exception as e:
        print(f"  Error: {e}")
        return
    
    # 测试2: Models
    print("\n[2] List Models...")
    try:
        resp = requests.get(f"{api_url}/v1/models", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Models: {resp.json()}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 测试3: Simple completion
    print("\n[3] Test Completion...")
    try:
        resp = requests.post(
            f"{api_url}/v1/completions",
            json={
                "model": "rwkv-7-world",
                "prompt": "The capital of France is",
                "max_tokens": 10
            },
            timeout=10
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Generated: {result['choices'][0]['text']}")
            print(f"  Tokens: {result['usage']['completion_tokens']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 测试4: Chat completion
    print("\n[4] Test Chat Completion...")
    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": "rwkv-7-world",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 20
            },
            timeout=10
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Response: {result['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAMBADA Evaluation via API")
    parser.add_argument("--api_url", type=str, default="http://10.100.1.98:16384", help="API server URL")
    parser.add_argument("--lambada_path", type=str, default="/public/home/ssjxzkz/Projects/rwkv-benchmark/eval/lambada_test.jsonl", help="Path to LAMBADA dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to test (default: all)")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of concurrent workers")
    parser.add_argument("--quick_test", action="store_true", help="Run quick API test only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test(args.api_url)
    else:
        test_lambada_via_api(
            api_url=args.api_url,
            lambada_path=args.lambada_path,
            max_samples=args.max_samples,
            max_workers=args.max_workers
        )

