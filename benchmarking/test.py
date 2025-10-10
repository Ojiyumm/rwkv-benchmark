"""
RWKV Batch Inference Tests
包含所有测试函数：Token分析、简单生成、随机问题、LAMBADA评估
"""

import numpy as np
import json
import random
import torch
from typing import List, Dict
from datetime import datetime
from torch.nn import functional as F

from batch_engine import RWKVInferenceEngine
import os


class BatchBenchmark:
    """
    批量推理基准测试
    负责数据集加载和测试执行
    """
    
    def __init__(self, engine: RWKVInferenceEngine):
        """
        初始化基准测试
        
        Args:
            engine: RWKV推理引擎实例
        """
        self.engine = engine
    
    def test_next_token_analysis(
        self,
        prompts: List[str],
        top_k: int = 5
    ):
        """
        测试1: 分析下一个token的概率分布
        
        Args:
            prompts: 提示列表
            top_k: 显示top-k个可能的token
        """
        print(f"\n{'='*80}")
        print(f"Test 1: Next Token Analysis")
        print(f"{'='*80}")
        
        self.engine.analyze_next_token(prompts, top_k=top_k)
        
        print(f"{'='*80}\n")
    
    def test_simple_generation(
        self,
        prompts: List[str],
        max_length: int = 10,
        noise: float = 0.0
    ) -> Dict:
        """
        测试2: 简单的生成测试
        
        Args:
            prompts: 提示列表
            max_length: 最大生成长度
            noise: 采样噪声
            
        Returns:
            包含结果和性能指标的字典
        """
        batch_size = len(prompts)
        
        print(f"\n{'='*80}")
        print(f"Test 2: Simple Generation")
        print(f"{'='*80}")
        print(f"Batch size: {batch_size}")
        print(f"Max length: {max_length}")
        print(f"Noise: {noise}")
        print()
        
        # 生成
        tokens, inference_time = self.engine.generate_batch(
            prompts,
            max_length=max_length,
            noise=noise
        )
        
        # 解码
        texts = self.engine.decode_tokens(tokens)
        
        # 计算性能指标
        total_tokens = tokens.size
        throughput = total_tokens / inference_time
        
        results = {
            'prompts': prompts,
            'generated_texts': texts,
            'tokens': tokens.tolist(),
            'batch_size': batch_size,
            'max_length': max_length,
            'total_tokens': total_tokens,
            'inference_time': inference_time,
            'throughput': throughput
        }
        
        # 打印结果
        print(f"Results:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Inference time: {inference_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} tokens/s")
        print()
        
        # 显示所有示例
        print("Generated outputs:")
        for i, (prompt, text) in enumerate(zip(prompts, texts)):
            print(f"\n  [{i}] {prompt}", end='')
            print(text)
            print(f"      {'#'*80}")
        
        print(f"\n{'='*80}\n")
        return results
    
    
    
    def test_lambada_eval(
        self,
        lambada_path: str = "/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
        batch_size: int = 256,
        print_interval: int = 1000
    ):
        """
        测试4: LAMBADA数据集评估（参考batch.py的实现）
        
        Args:
            lambada_path: LAMBADA数据集路径
            batch_size: 批量大小
            print_interval: 打印间隔
        """
        print(f"\n{'='*80}")
        print(f"Test 4: LAMBADA Evaluation")
        print(f"{'='*80}")
        print(f"Batch size: {batch_size}")
        print(f"Dataset: {lambada_path}")
        print()
        
        # 加载数据集
        try:
            with open(lambada_path, "r", encoding="utf-8") as f:
                todo = [json.loads(line) for line in f]
                todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]
        except FileNotFoundError:
            print(f"Error: LAMBADA dataset not found at {lambada_path}")
            print("Please download it first or skip this test.")
            print(f"{'='*80}\n")
            return
        
        print(f"Loaded {len(todo)} examples")
        print("Starting evaluation...\n")
        
        # 评估
        self._eval_qa_batch(todo, print_interval, batch_size=batch_size)
        
        print(f"\n{'='*80}\n")
    
    def _eval_qa_batch(
        self,
        todo: List[List[str]],
        print_interval: int,
        pad_eod: bool = True,
        batch_size: int = 1
    ):
        """
        批量问答评估（参考batch.py的实现）
        
        Args:
            todo: 问答对列表
            print_interval: 打印间隔
            pad_eod: 是否在开头添加EOD token
            batch_size: 批量大小
        """
        xsum = 0
        xcnt = 0
        xacc = 0
        
        fwd_tokens = []
        fwd_desc = []
        
        for i in range(len(todo)):
            # get src and dst
            d = todo[i]
            if pad_eod:
                src = [0] + self.engine.tokenizer.encode(d[0])
            else:
                src = self.engine.tokenizer.encode(d[0])
            dst = self.engine.tokenizer.encode(d[1])
            
            # store jobs
            fwd_tokens.append(src + dst)
            fwd_desc.append((src, dst))
            
            if len(fwd_tokens) >= batch_size or i == len(todo) - 1:
                # batch forward
                out_batch = self.engine.model.forward_batch(
                    fwd_tokens,
                    self.engine.model.generate_zero_state(len(fwd_tokens)),
                    full_output=True
                )
                
                # process output
                for j in range(len(fwd_desc)):
                    out = out_batch[j]
                    src, dst = fwd_desc[j]
                    
                    logits = 0
                    correct = True
                    for n in range(len(dst)):
                        ooo = out[len(src) - 1 + n].float()
                        probs = F.softmax(ooo, dim=-1)
                        logits += np.log(probs[dst[n]].item())
                        if torch.argmax(probs).item() != dst[n]:
                            correct = False
                    
                    xcnt += 1
                    xsum += logits
                    xacc += 1 if correct else 0
                    
                    if xcnt % print_interval == 0 or xcnt == len(todo):
                        ppl = np.exp(-xsum / xcnt)
                        acc = xacc / xcnt * 100
                        print(f"{xcnt:5d} samples | ppl {ppl:.2f} | acc {acc:.1f}%")
                
                fwd_tokens = []
                fwd_desc = []


def test_all():
    """运行所有测试"""
    print(f"\n{'='*80}")
    print(f"RWKV Batch Inference Engine - Complete Test Suite")
    print(f"{'='*80}")
    
    # 从环境变量获取模型路径
    model_path = os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096")
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print()
    
    # 初始化引擎
    print("Initializing RWKV Inference Engine...")
    engine = RWKVInferenceEngine(
        model_path=model_path,
        seed=42
    )
    
    # 创建基准测试
    benchmark = BatchBenchmark(engine)
    
    # 测试1: 分析下一个token
    test_prompts = [
        "The apple can be",
        "The cat can't be",
        "他们发现，这",
        "Q: 1+1=?\nA: 1+1=2."
    ]
    benchmark.test_next_token_analysis(test_prompts, top_k=5)
    
    # 测试2: 简单生成
    simple_prompts = [
        "也许",
        "我看到",
        "他们发现",
        "我认为",
        "哈哈",
        "这是一个有趣的",
        "List of Emojis:"
    ]
    benchmark.test_simple_generation(
        prompts=simple_prompts,
        max_length=10,
        noise=0.0
    )
    
    # 测试3: 随机问题（主要测试）
    benchmark.test_random_questions(
        num_samples=128,
        max_length=150,
        noise=3.0,
        save_jsonl=True
    )
    
    # 测试4: LAMBADA评估
    benchmark.test_lambada_eval(batch_size=256)
    
    print(f"\n{'='*80}")
    print(f"All tests completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_all()

