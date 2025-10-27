"""
RWKV Batch Inference Engine
只负责模型加载、批量推理和解码
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types
import torch
import time
import random
import sys
import os
import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from torch.nn import functional as F

# Add parent directory to path to import reference modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch


class RWKVInferenceEngine:
    """
    RWKV推理引擎
    负责模型加载和批量推理
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        vocab_size: int = 65536,
        head_size: int = 64,
        seed: int = 42
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径
            vocab_path: 词汇表路径
            vocab_size: 词汇表大小
            head_size: 注意力头大小
            seed: 随机种子
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # 初始化模型参数
        args = types.SimpleNamespace()
        args.vocab_size = vocab_size
        args.head_size = head_size
        args.MODEL_NAME = model_path
        
        print(f'\nLoading RWKV model from {model_path}...')
        self.model = RWKV_x070(args)
        print(f'Model loaded successfully!\n')
        
        # 初始化tokenizer
        if vocab_path is None:
            vocab_path = os.path.join(os.path.dirname(__file__), "..", "reference", "rwkv_vocab_v20230424.txt")
        self.tokenizer = TRIE_TOKENIZER(vocab_path)
        
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 100,
        noise: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        批量生成文本
        
        Args:
            prompts: 提示列表
            max_length: 最大生成长度
            noise: 采样噪声
            
        Returns:
            tokens: numpy数组，形状为 (batch_size, max_length)
            inference_time: 推理总时间（秒）
        """
        batch_size = len(prompts)
        print(f"    [BatchEngine] generate_batch called with batch_size={batch_size}, max_length={max_length}")
        
        # 初始化状态
        state = self.model.generate_zero_state(batch_size)
        
        # Prefill阶段
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        out = self.model.forward_batch(
            [self.tokenizer.encode(prompt) for prompt in prompts],
            state
        )
        
        # Decode阶段 - 参考batch.py的实现
        tokens = []
        finished = [False] * batch_size  # 追踪每个序列是否完成
        END_TOKEN_ID = 0  # <|endoftext|> token ID
        
        for i in range(max_length):
            # Sample - 返回形状为 (batch_size, 1) 的tensor
            new_tokens = sampler_simple_batch(out, noise=noise).tolist()
            
            # 检查停止标记：如果序列已完成，用 END_TOKEN_ID 填充
            for b_idx in range(batch_size):
                if finished[b_idx]:
                    new_tokens[b_idx][0] = END_TOKEN_ID
                elif new_tokens[b_idx][0] == END_TOKEN_ID:
                    finished[b_idx] = True
            
            tokens.append(new_tokens)
            
            # 如果所有序列都完成，提前退出
            if all(finished):
                print(f"    [BatchEngine] All sequences finished at step {i+1}/{max_length}")
                break
            
            # Forward
            out = self.model.forward_batch(new_tokens, state)
        
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        
        # 如果提前停止，用 END_TOKEN_ID 填充剩余位置
        actual_length = len(tokens)
        if actual_length < max_length:
            # 每个时间步需要 batch_size 个 [token]
            for _ in range(max_length - actual_length):
                tokens.append([[END_TOKEN_ID] for _ in range(batch_size)])
        
        # 转换为numpy数组并整理维度
        # tokens: list of [batch_size, 1] -> (max_length, batch_size, 1)
        # transpose -> (batch_size, max_length, 1)
        # squeeze(-1) -> (batch_size, max_length)
        tokens = np.transpose(np.array(tokens), axes=(1, 0, 2)).squeeze(-1)
        
        return tokens, inference_time
    
    def decode_tokens(
        self,
        tokens: np.ndarray,
        utf8_errors: str = "ignore"
    ) -> List[str]:
        """
        将token数组解码为文本
        
        Args:
            tokens: numpy数组，形状为 (batch_size, seq_length)
            utf8_errors: UTF-8错误处理方式
            
        Returns:
            解码后的文本列表
        """
        texts = []
        for token_seq in tokens:
            try:
                text = self.tokenizer.decode(token_seq, utf8_errors=utf8_errors)
                texts.append(text)
            except Exception as e:
                print(f"Warning: Failed to decode: {e}")
                texts.append("")
        return texts
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        max_length: int = 100,
        echo: bool = False,
        top_logprobs: int = 1
    ):
        """
        批量生成文本并返回logprobs信息（参考batch.py的高效batch实现）
        支持多个prompt的batch推理（变长序列）
        
        Args:
            prompts: 输入提示列表
            max_length: 最大生成长度  
            echo: 是否包含prompt的logprobs
            top_logprobs: 返回top-N logprobs
            
        Returns:
            字典列表，每个包含tokens, logprobs等信息
        """
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        
        # Encode所有prompts
        prompt_tokens_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        
        results = []
        
        # 如果echo=True，使用full_output=True一次性forward所有prompts（变长batch）
        if echo:
            # 使用batch forward，支持变长序列
            full_outputs = self.model.forward_batch(prompt_tokens_list, state, full_output=True)
            
            for batch_idx in range(batch_size):
                prompt_tokens = prompt_tokens_list[batch_idx]
                full_output = full_outputs[batch_idx]
                
                all_tokens = []
                all_logprobs = []
                all_top_logprobs = []
                
                # 对prompt tokens计算logprobs
                for i in range(len(prompt_tokens)):
                    token = prompt_tokens[i]
                    all_tokens.append(token)
                    
                    if i == 0:
                        # 第一个token，没有前序预测，logprob设为0
                        all_logprobs.append(0.0)
                        all_top_logprobs.append({})
                    else:
                        # 使用前一个位置的输出来预测当前token
                        logits = full_output[i-1].float()
                        log_probs = F.log_softmax(logits, dim=-1)
                        
                        token_logprob = log_probs[token].item()
                        all_logprobs.append(token_logprob)
                        
                        # Top logprobs
                        top_lp, top_idx = torch.topk(log_probs, min(top_logprobs, len(log_probs)))
                        top_dict = {}
                        for lp, ti in zip(top_lp.tolist(), top_idx.tolist()):
                            token_str = self.tokenizer.decode([ti], utf8_errors="ignore")
                            top_dict[token_str] = lp
                        all_top_logprobs.append(top_dict)
                
                # 准备生成的初始输出（最后一个位置）
                results.append({
                    'all_tokens': all_tokens,
                    'all_logprobs': all_logprobs,
                    'all_top_logprobs': all_top_logprobs,
                    'last_out': full_output[-1]  # 用于继续生成
                })
        else:
            # 不echo，只forward prompts不记录logprobs
            outs = self.model.forward_batch(prompt_tokens_list, state)
            for batch_idx in range(batch_size):
                results.append({
                    'all_tokens': [],
                    'all_logprobs': [],
                    'all_top_logprobs': [],
                    'last_out': outs[batch_idx]
                })
        
        # Generate new tokens（batch方式）
        for _ in range(max_length):
            # 收集当前batch所有的输出
            batch_outs = torch.stack([r['last_out'] for r in results])
            log_probs_batch = F.log_softmax(batch_outs.float(), dim=-1)
            
            # Sample new tokens for all in batch
            new_tokens = sampler_simple_batch(batch_outs, noise=0.0).tolist()
            
            # 记录每个样本的token和logprob
            for batch_idx in range(batch_size):
                new_token = new_tokens[batch_idx][0]
                log_probs = log_probs_batch[batch_idx]
                token_logprob = log_probs[new_token].item()
                
                results[batch_idx]['all_tokens'].append(new_token)
                results[batch_idx]['all_logprobs'].append(token_logprob)
                
                # Top logprobs
                top_lp, top_idx = torch.topk(log_probs, min(top_logprobs, len(log_probs)))
                top_dict = {}
                for lp, ti in zip(top_lp.tolist(), top_idx.tolist()):
                    token_str = self.tokenizer.decode([ti], utf8_errors="ignore")
                    top_dict[token_str] = lp
                results[batch_idx]['all_top_logprobs'].append(top_dict)
            
            # Batch forward
            new_tokens_for_forward = [[t[0]] for t in new_tokens]
            outs = self.model.forward_batch(new_tokens_for_forward, state)
            for batch_idx in range(batch_size):
                results[batch_idx]['last_out'] = outs[batch_idx]
        
        # 最终格式化输出
        final_results = []
        for batch_idx in range(batch_size):
            all_tokens = results[batch_idx]['all_tokens']
            all_logprobs = results[batch_idx]['all_logprobs']
            all_top_logprobs = results[batch_idx]['all_top_logprobs']
            
            # Decode all tokens
            token_strs = []
            for token in all_tokens:
                token_str = self.tokenizer.decode([token], utf8_errors="ignore")
                token_strs.append(token_str)
            
            final_results.append({
                'tokens': all_tokens,
                'token_strs': token_strs,
                'logprobs': all_logprobs,
                'top_logprobs': all_top_logprobs,
                'text': ''.join(token_strs)
            })
        
        return final_results
    
    def analyze_next_token(
        self,
        prompts: List[str],
        top_k: int = 5
    ):
        """
        分析每个prompt的下一个token概率分布
        
        Args:
            prompts: 提示列表
            top_k: 显示top-k个可能的token
        """
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        outs = self.model.forward_batch(
            [self.tokenizer.encode(prompt) for prompt in prompts],
            state
        )
        
        for n in range(batch_size):
            print(f"\nPrompt: {prompts[n]}")
            out = outs[n]
            probs = F.softmax(out.float(), dim=-1)
            _, indices = torch.topk(probs, top_k)
            
            for i in range(len(indices)):
                token_id = indices[i].item()
                token = self.tokenizer.decode([token_id])
                token_prob = probs[token_id].item()
                print(f"  {repr(token):<20} [probability {token_prob:.2%}]")
    
    def save_qa_to_jsonl(
        self,
        prompts: List[str],
        responses: List[str],
        task_name: str,
        output_dir: str = ".",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存问答对到jsonl文件
        
        Args:
            prompts: 问题列表
            responses: 回答列表
            task_name: 任务名称（用于文件命名）
            output_dir: 输出目录
            metadata: 额外的元数据（可选）
            
        Returns:
            保存的文件路径
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Prompts and responses must have the same length: {len(prompts)} vs {len(responses)}")
        
        # 生成文件名：task_name_timestamp.jsonl
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入jsonl文件
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                record = {
                    "index": i,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 添加元数据
                if metadata:
                    record.update(metadata)
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(prompts)} Q&A pairs to {filepath}")
        return filepath
    
    def generate_and_save(
        self,
        prompts: List[str],
        task_name: str,
        max_length: int = 100,
        output_dir: str = "/home/rwkv/Peter/rwkveval",
        save_tokens: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], str]:
        """
        生成回答并保存到jsonl文件（便捷方法）
        
        Args:
            prompts: 问题列表
            task_name: 任务名称
            max_length: 最大生成长度
            output_dir: 输出目录
            save_tokens: 是否保存token信息
            metadata: 额外的元数据
            
        Returns:
            (responses, filepath): 回答列表和保存的文件路径
        """
        print(f"Generating {len(prompts)} responses for task: {task_name}")
        
        # 生成回答
        start_time = time.time()
        tokens, inference_time = self.generate_batch(
            prompts=prompts,
            max_length=max_length,
            noise=0.0
        )
        responses = self.decode_tokens(tokens)
        
        total_time = time.time() - start_time
        
        # 准备元数据
        task_metadata = {
            "task": task_name,
            "max_length": max_length,
            "num_samples": len(prompts),
            "inference_time": round(inference_time, 4),
            "total_time": round(total_time, 4),
            "throughput": round(len(prompts) / total_time, 2)
        }
        
        if metadata:
            task_metadata.update(metadata)
        
        # 如果需要保存tokens
        if save_tokens:
            # 生成带token信息的文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_name}_{timestamp}.jsonl"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, (prompt, response, token_seq) in enumerate(zip(prompts, responses, tokens)):
                    record = {
                        "index": i,
                        "prompt": prompt,
                        "response": response,
                        "tokens": token_seq.tolist(),
                        "num_tokens": len(token_seq),
                        "timestamp": datetime.now().isoformat()
                    }
                    record.update(task_metadata)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        else:
            # 只保存文本
            filepath = self.save_qa_to_jsonl(
                prompts=prompts,
                responses=responses,
                task_name=task_name,
                output_dir=output_dir,
                metadata=task_metadata
            )
        
        print(f"Task '{task_name}' completed in {total_time:.2f}s")
        print(f"Throughput: {len(prompts)/total_time:.2f} samples/s")
        
        return responses, filepath


if __name__ == "__main__":
    # 简单测试：加载模型并生成
    import sys
    
    print("Testing RWKV Inference Engine...")
    
    # 从环境变量或命令行参数获取模型路径
    model_path = os.getenv("MODEL_PATH", "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"Model path: {model_path}")
    
    engine = RWKVInferenceEngine(
        model_path=model_path,
        seed=42
    )
    
    # 测试生成
    prompts = ["The", "Hello", "今天天气"]
    print(f"\nTesting batch generation with {len(prompts)} prompts...")
    tokens, inference_time = engine.generate_batch(prompts, max_length=20, noise=0.0)
    texts = engine.decode_tokens(tokens)
    
    print(f"\nResults:")
    print(f"  Inference time: {inference_time:.4f}s")
    print(f"  Throughput: {tokens.size/inference_time:.2f} tokens/s")
    
    for prompt, text in zip(prompts, texts):
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {text}")
    
    # 测试保存功能
    print(f"\n{'='*80}")
    print("Testing save_qa_to_jsonl...")
    print(f"{'='*80}")
    
    test_prompts = ["The capital of France is", "1+1=", "今天天气"]
    responses, saved_file = engine.generate_and_save(
        prompts=test_prompts,
        task_name="test_generation",
        max_length=10,
        output_dir="./results",
        save_tokens=False,
        metadata={"model": "rwkv7-0.4b", "test": True}
    )
    
    print(f"\nSaved to: {saved_file}")
    print(f"Generated {len(responses)} responses")
    
    print("\nEngine test completed! Use test.py for full test suite.")

