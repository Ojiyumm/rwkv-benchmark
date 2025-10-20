"""
注册 ASDiv 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry


def load_asdiv(path: str) -> List[Dict]:
    """
    加载 ASDiv 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 ASDiv 数据")
    return data


def asdiv_prompt(item: Dict) -> str:
    """
    ASDiv prompt template
    """
    body = item.get("body")
    question = item.get("question", "")
    context = f"{body}\n\n" if body else ""
    return (
        "You are a very talented math tutor who solves applied word problems meticulously.\n"
        "Reason step by step inside <think>...</think> before providing the final numerical answer.\n\n"
        f"{context}"
        f"Question: {question}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="asdiv",
    loader=load_asdiv,
    prompt_template=asdiv_prompt,
    description="ASDiv math word problems (short-form answers)",
    default_batch_size=64,
    default_max_length=64,
)
