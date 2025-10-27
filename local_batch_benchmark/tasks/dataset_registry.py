"""
Dataset Registry - 数据集注册器
支持注册不同的数据集和对应的 prompt template
"""

import json
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    loader: Callable  # 数据加载函数
    prompt_template: Callable  # prompt 模板函数
    description: str = ""
    default_batch_size: int = 128
    default_max_length: int = 100


class DatasetRegistry:
    """数据集注册器"""
    
    _registry: Dict[str, DatasetConfig] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        loader: Callable,
        prompt_template: Callable,
        description: str = "",
        default_batch_size: int = 128,
        default_max_length: int = 100
    ):
        """
        注册数据集
        
        Args:
            name: 数据集名称
            loader: 数据加载函数 (path: str) -> List[Dict]
            prompt_template: prompt模板函数 (data: Dict) -> str
            description: 数据集描述
            default_batch_size: 默认批量大小
            default_max_length: 默认最大生成长度
        """
        config = DatasetConfig(
            name=name,
            loader=loader,
            prompt_template=prompt_template,
            description=description,
            default_batch_size=default_batch_size,
            default_max_length=default_max_length
        )
        cls._registry[name] = config
        print(f"✓ Registered dataset: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[DatasetConfig]:
        """获取数据集配置"""
        return cls._registry.get(name)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """列出所有已注册的数据集"""
        return list(cls._registry.keys())
    
    @classmethod
    def load_dataset(
        cls,
        name: str,
        path: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Args:
            name: 数据集名称
            path: 数据集路径
            limit: 限制数据量（用于快速测试）
            
        Returns:
            数据列表，每个元素包含 'prompt' 和 'reference' 字段
        """
        config = cls.get(name)
        if config is None:
            raise ValueError(f"Dataset '{name}' not registered. Available: {cls.list_datasets()}")
        
        # 加载原始数据
        raw_data = config.loader(path)
        
        # 应用 prompt template
        processed_data = []
        for item in raw_data[:limit] if limit else raw_data:
            prompt = config.prompt_template(item)
            # 提取答案（不同数据集可能用 'answer' 或 'reference' 字段）
            reference = item.get('answer') or item.get('reference', '')
            processed_data.append({
                'prompt': prompt,
                'reference': reference,
                'raw': item
            })
        
        print(f"✓ Loaded {len(processed_data)} samples from {name}")
        return processed_data


# ==================== 预定义数据集加载器 ====================

def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 格式数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_json(path: str) -> List[Dict]:
    """加载 JSON 格式数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 如果是字典，尝试获取数据列表
        if isinstance(data, dict):
            return data.get('data', data.get('examples', [data]))
        return data


# ==================== 预定义 Prompt Templates ====================

def lambada_prompt(item: Dict) -> str:
    """LAMBADA 数据集的 prompt template"""
    text = item.get('text', '')
    # 移除最后一个单词作为预测目标
    return text.rsplit(' ', 1)[0]


def qa_prompt(item: Dict) -> str:
    """问答数据集的 prompt template"""
    question = item.get('question', '')
    return f"Q: {question}\nA:"


def completion_prompt(item: Dict) -> str:
    """补全任务的 prompt template"""
    return item.get('prompt', item.get('text', ''))


def cot_prompt(item: Dict) -> str:
    """Chain-of-Thought 推理的 prompt template"""
    question = item.get('question', '')
    return f"{question}\nLet's think step by step:\n"


def math_prompt(item: Dict) -> str:
    """数学问题的 prompt template"""
    problem = item.get('problem', item.get('question', ''))
    return f"Problem: {problem}\n\nSolution:\n"


# ==================== 注册常见数据集 ====================

# LAMBADA
DatasetRegistry.register(
    name='lambada',
    loader=load_jsonl,
    prompt_template=lambada_prompt,
    description='LAMBADA language modeling benchmark',
    default_batch_size=64,  # 降低，因为 perplexity 用 full_output 会占用大量显存
    default_max_length=1
)

# 通用问答
DatasetRegistry.register(
    name='qa',
    loader=load_jsonl,
    prompt_template=qa_prompt,
    description='General question answering',
    default_batch_size=128,
    default_max_length=100
)

# 补全任务
DatasetRegistry.register(
    name='completion',
    loader=load_jsonl,
    prompt_template=completion_prompt,
    description='Text completion task',
    default_batch_size=128,
    default_max_length=100
)

# Chain-of-Thought
DatasetRegistry.register(
    name='cot',
    loader=load_json,
    prompt_template=cot_prompt,
    description='Chain-of-thought reasoning',
    default_batch_size=64,
    default_max_length=256
)

# 数学问题
DatasetRegistry.register(
    name='math',
    loader=load_json,
    prompt_template=math_prompt,
    description='Math problem solving',
    default_batch_size=64,
    default_max_length=256
)


if __name__ == "__main__":
    # 测试注册器
    print("\n=== Dataset Registry Test ===\n")
    print(f"Registered datasets: {DatasetRegistry.list_datasets()}")
    
    for name in DatasetRegistry.list_datasets():
        config = DatasetRegistry.get(name)
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Default batch size: {config.default_batch_size}")
        print(f"  Default max length: {config.default_max_length}")

