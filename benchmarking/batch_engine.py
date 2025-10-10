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
from typing import List, Optional, Tuple
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
        for i in range(max_length):
            # Sample - 返回形状为 (batch_size, 1) 的tensor
            new_tokens = sampler_simple_batch(out, noise=noise).tolist()
            tokens.append(new_tokens)
            # Forward
            out = self.model.forward_batch(new_tokens, state)
        
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        
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
        生成文本并返回logprobs信息（参考batch.py的高效实现）
        
        Args:
            prompts: 输入提示列表
            max_length: 最大生成长度  
            echo: 是否包含prompt的logprobs
            top_logprobs: 返回top-N logprobs
            
        Returns:
            字典，包含tokens, logprobs等信息
        """
        # 注意：这个方法只处理单个prompt（batch_size=1）
        if len(prompts) != 1:
            raise ValueError("generate_with_logprobs only supports batch_size=1")
        
        prompt = prompts[0]
        prompt_tokens = self.tokenizer.encode(prompt)
        
        # 初始化 batch_size=1 的state
        state = self.model.generate_zero_state(1)
        
        all_tokens = []
        all_logprobs = []
        all_top_logprobs = []
        
        # 如果echo=True，需要包含prompt的logprobs
        # 使用 full_output=True 一次性获取所有位置的输出（参考batch.py）
        if echo and len(prompt_tokens) > 0:
            # Forward prompt + 生成，使用full_output=True
            # 先forward prompt获取所有位置输出
            full_output = self.model.forward_batch([prompt_tokens], state, full_output=True)[0]
            
            # 对prompt tokens计算logprobs（从第2个token开始，因为第1个token没有"前一个输出"）
            for i in range(len(prompt_tokens)):
                token = prompt_tokens[i]
                all_tokens.append(token)
                
                if i == 0:
                    # 第一个token，没有前序预测，logprob设为0（OpenAI的做法）
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
            
            # 最后一个位置的输出用于生成第一个新token
            out = full_output[-1:, :]  # 保持2D形状
        else:
            # 不echo，只forward prompt
            if len(prompt_tokens) > 0:
                out = self.model.forward_batch([prompt_tokens], state)
            else:
                # 空prompt
                out = self.model.forward_batch([[0]], state)
        
        # Generate new tokens
        for _ in range(max_length):
            logits = out[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            
            # Sample next token
            new_token = sampler_simple_batch(out, noise=0.0)[0][0].item()
            token_logprob = log_probs[new_token].item()
            
            all_tokens.append(new_token)
            all_logprobs.append(token_logprob)
            
            # Top logprobs
            top_lp, top_idx = torch.topk(log_probs, min(top_logprobs, len(log_probs)))
            top_dict = {}
            for lp, ti in zip(top_lp.tolist(), top_idx.tolist()):
                token_str = self.tokenizer.decode([ti], utf8_errors="ignore")
                top_dict[token_str] = lp
            all_top_logprobs.append(top_dict)
            
            # Forward
            out = self.model.forward_batch([[new_token]], state)
        
        # Decode all tokens
        token_strs = []
        for token in all_tokens:
            token_str = self.tokenizer.decode([token], utf8_errors="ignore")
            token_strs.append(token_str)
        
        return [{
            'tokens': all_tokens,
            'token_strs': token_strs,
            'logprobs': all_logprobs,
            'top_logprobs': all_top_logprobs,
            'text': ''.join(token_strs)
        }]
    
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
    
    print("\nEngine test completed! Use test.py for full test suite.")

