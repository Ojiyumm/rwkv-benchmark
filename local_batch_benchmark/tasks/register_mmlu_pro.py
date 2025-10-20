"""
注册 MMLU Pro 数据集
示例：如何添加新的测评任务
"""

import json
from typing import List, Dict
from .dataset_registry import DatasetRegistry


# ==================== 数据加载器 ====================

def load_mmlu_pro(path: str) -> List[Dict]:
    """
    加载 MMLU Pro 数据集（test split）
    """
    from datasets import load_dataset
    
    # 直接加载 test split
    ds = load_dataset('parquet', data_dir=path, split='test')
    
    # 转换为列表
    data = list(ds)
    
    print(f"  ✓ 加载了 {len(data)} 条 test 数据")
    return data


# ==================== Prompt Templates ====================

def _parse_options(options) -> List[str]:
    
    if isinstance(options, list):
        # 过滤掉 None 和空字符串
        return [opt.strip() for opt in options if opt and str(opt).strip()]
    elif isinstance(options, str):
        # 如果是逗号分隔的字符串，按逗号分割并过滤空项
        return [opt.strip() for opt in options.split(',') if opt.strip()]
    else:
        return []


def mmlu_pro_prompt(item: Dict) -> str:
    question = item.get('question', '')
    options_raw = item.get('options', item.get('choices', []))
    category = item.get('category', item.get('subject', 'professional knowledge'))
    
    # 解析并过滤选项（移除空项）
    options = _parse_options(options_raw)
    
    # 构建选项字符串（只为实际存在的选项分配字母）
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    option_text = '\n'.join([
        f"{option_labels[i]}. {opt}"
        for i, opt in enumerate(options) if i < len(option_labels)
    ])
    
    role_line = f"You are a very talented expert in {category}."
    
    return (
        f"{role_line}\n"
        "Carefully study the professional multiple-choice question and pick the single best option.\n"
        "Respond using this format:\n"
        "<think>your step-by-step reasoning</think>\n"
        "Final Answer: <option letter>\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"{option_text}\n\n"
        "Begin your reasoning now.\n<think>"
    )


def mmlu_pro_prompt_cot(item: Dict) -> str:
    """
    MMLU Pro 的 Chain-of-Thought prompt template
    自动将 options 列表标注为 A、B、C、D 等
    
    注意：
    - 只为非空的有效选项分配字母标签
    - 如果有4个选项，输出 A、B、C、D
    - 如果有10个选项，输出 A、B、C、D、E、F、G、H、I、J
    - 不会输出内容为空的选项
    """
    question = item.get('question', '')
    options_raw = item.get('options', item.get('choices', []))
    
    # 解析并过滤选项（移除空项）
    options = _parse_options(options_raw)
    
    # 构建选项字符串（只为实际存在的选项分配字母）
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    option_text = '\n'.join([
        f"{option_labels[i]}. {opt}" 
        for i, opt in enumerate(options) if i < len(option_labels)
    ])
    
    category = item.get('category', item.get('subject', 'professional knowledge'))
    
    return (
        f"You are a very talented expert in {category}.\n"
        "Conduct a careful chain-of-thought analysis before giving the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Final Answer: <option letter>\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"{option_text}\n\n"
        "Begin your reasoning now.\n<think>"
    )


# ==================== 注册 ====================

# 注册标准版本
DatasetRegistry.register(
    name='mmlu_pro',
    loader=load_mmlu_pro,
    prompt_template=mmlu_pro_prompt,
    description='MMLU Pro - Multi-task Language Understanding (Professional)',
    default_batch_size=64,
    default_max_length=10  # 多选题只需要生成答案字母
)

# 注册 CoT 版本
DatasetRegistry.register(
    name='mmlu_pro_cot',
    loader=load_mmlu_pro,
    prompt_template=mmlu_pro_prompt_cot,
    description='MMLU Pro with Chain-of-Thought',
    default_batch_size=32,
    default_max_length=200  # CoT 需要更长的生成长度
)

