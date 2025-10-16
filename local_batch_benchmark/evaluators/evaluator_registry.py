"""
Evaluator Registry - 评估器注册器
支持注册不同的评估方法和指标计算
"""

import numpy as np
import torch
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    name: str
    evaluator: Callable  # 评估函数
    description: str = ""
    metrics: List[str] = None  # 输出的指标名称


class EvaluatorRegistry:
    """评估器注册器"""
    
    _registry: Dict[str, EvaluatorConfig] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        evaluator: Callable,
        description: str = "",
        metrics: List[str] = None
    ):
        """
        注册评估器
        
        Args:
            name: 评估器名称
            evaluator: 评估函数 (engine, data, **kwargs) -> Dict[str, float]
            description: 评估器描述
            metrics: 输出的指标名称列表
        """
        config = EvaluatorConfig(
            name=name,
            evaluator=evaluator,
            description=description,
            metrics=metrics or []
        )
        cls._registry[name] = config
        print(f"✓ Registered evaluator: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[EvaluatorConfig]:
        """获取评估器配置"""
        return cls._registry.get(name)
    
    @classmethod
    def list_evaluators(cls) -> List[str]:
        """列出所有已注册的评估器"""
        return list(cls._registry.keys())
    
    @classmethod
    def evaluate(
        cls,
        name: str,
        engine,
        data: List[Dict],
        **kwargs
    ) -> Dict[str, float]:
        """
        运行评估
        
        Args:
            name: 评估器名称
            engine: 推理引擎
            data: 数据列表
            **kwargs: 额外参数
            
        Returns:
            评估结果字典
        """
        config = cls.get(name)
        if config is None:
            raise ValueError(f"Evaluator '{name}' not registered. Available: {cls.list_evaluators()}")
        
        return config.evaluator(engine, data, **kwargs)


# ==================== 预定义评估器 ====================

def exact_match_evaluator(
    engine,
    data: List[Dict],
    batch_size: int = 128,
    max_length: int = 100,
    noise: float = 0.0,
    **kwargs
) -> Dict[str, float]:
    """
    精确匹配评估器
    适用于分类、简单QA等任务
    """
    print(f"\n=== Running Exact Match Evaluation ===")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {batch_size}\n")
    
    correct = 0
    total = 0
    
    # 批量处理
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        references = [item['reference'] for item in batch]
        
        # 生成
        tokens, _ = engine.generate_batch(prompts, max_length=max_length, noise=noise)
        predictions = engine.decode_tokens(tokens)
        
        # 评估
        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip().lower()
            ref_clean = ref.strip().lower()
            if pred_clean == ref_clean:
                correct += 1
            total += 1
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {total}/{len(data)} samples, accuracy: {correct/total*100:.2f}%")
    
    accuracy = correct / total if total > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
    return results


def perplexity_evaluator(
    engine,
    data: List[Dict],
    batch_size: int = 256,
    pad_eod: bool = True,
    print_interval: int = 1000,
    **kwargs
) -> Dict[str, float]:
    """
    困惑度评估器
    适用于语言建模任务（如LAMBADA）
    """
    print(f"\n=== Running Perplexity Evaluation ===")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {batch_size}\n")
    
    xsum = 0
    xcnt = 0
    xacc = 0
    
    fwd_tokens = []
    fwd_desc = []
    
    for i in range(len(data)):
        item = data[i]
        
        # 获取 prompt 和 reference
        if 'raw' in item and 'text' in item['raw']:
            # LAMBADA 格式
            text = item['raw']['text']
            parts = text.rsplit(' ', 1)
            src_text = parts[0]
            dst_text = ' ' + parts[1] if len(parts) > 1 else ''
        else:
            src_text = item['prompt']
            dst_text = item['reference']
        
        # 编码
        if pad_eod:
            src = [0] + engine.tokenizer.encode(src_text)
        else:
            src = engine.tokenizer.encode(src_text)
        dst = engine.tokenizer.encode(dst_text)
        
        # 存储任务
        fwd_tokens.append(src + dst)
        fwd_desc.append((src, dst))
        
        # 批量前向
        if len(fwd_tokens) >= batch_size or i == len(data) - 1:
            out_batch = engine.model.forward_batch(
                fwd_tokens,
                engine.model.generate_zero_state(len(fwd_tokens)),
                full_output=True
            )
            
            # 处理输出
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
                
                if xcnt % print_interval == 0 or xcnt == len(data):
                    ppl = np.exp(-xsum / xcnt)
                    acc = xacc / xcnt * 100
                    print(f"{xcnt:5d} samples | ppl {ppl:.2f} | acc {acc:.1f}%")
            
            fwd_tokens = []
            fwd_desc = []
    
    ppl = np.exp(-xsum / xcnt) if xcnt > 0 else float('inf')
    acc = xacc / xcnt if xcnt > 0 else 0
    
    results = {
        'perplexity': float(ppl),
        'accuracy': float(acc),
        'total': xcnt
    }
    
    print(f"\nFinal Results:")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Accuracy: {acc*100:.1f}%")
    
    return results


def generation_evaluator(
    engine,
    data: List[Dict],
    batch_size: int = 128,
    max_length: int = 100,
    noise: float = 0.0,
    save_outputs: bool = True,
    output_file: str = None,
    **kwargs
) -> Dict[str, float]:
    """
    生成评估器
    生成文本并计算基本统计信息
    """
    print(f"\n=== Running Generation Evaluation ===")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}\n")
    
    all_outputs = []
    total_tokens = 0
    total_time = 0
    
    # 批量处理
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 生成
        tokens, inference_time = engine.generate_batch(
            prompts,
            max_length=max_length,
            noise=noise
        )
        predictions = engine.decode_tokens(tokens)
        
        # 收集结果
        for j, pred in enumerate(predictions):
            all_outputs.append({
                'prompt': prompts[j],
                'generated': pred,
                'reference': batch[j].get('reference', '')
            })
        
        total_tokens += tokens.size
        total_time += inference_time
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i+batch_size, len(data))}/{len(data)} samples")
    
    # 保存输出
    if save_outputs:
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"generation_outputs_{timestamp}.jsonl"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_outputs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n✓ Saved outputs to {output_file}")
    
    # 计算统计
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_length = total_tokens / len(data) if len(data) > 0 else 0
    
    results = {
        'total_samples': len(data),
        'total_tokens': total_tokens,
        'total_time': total_time,
        'throughput': throughput,
        'avg_length': avg_length
    }
    
    print(f"\nStatistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Avg length: {avg_length:.1f} tokens/sample")
    
    return results


# ==================== 注册评估器 ====================

EvaluatorRegistry.register(
    name='exact_match',
    evaluator=exact_match_evaluator,
    description='Exact match accuracy',
    metrics=['accuracy', 'correct', 'total']
)

EvaluatorRegistry.register(
    name='perplexity',
    evaluator=perplexity_evaluator,
    description='Perplexity and accuracy (for language modeling)',
    metrics=['perplexity', 'accuracy', 'total']
)

EvaluatorRegistry.register(
    name='generation',
    evaluator=generation_evaluator,
    description='Text generation with statistics',
    metrics=['total_samples', 'total_tokens', 'throughput', 'avg_length']
)


if __name__ == "__main__":
    # 测试注册器
    print("\n=== Evaluator Registry Test ===\n")
    print(f"Registered evaluators: {EvaluatorRegistry.list_evaluators()}")
    
    for name in EvaluatorRegistry.list_evaluators():
        config = EvaluatorRegistry.get(name)
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Metrics: {', '.join(config.metrics)}")

