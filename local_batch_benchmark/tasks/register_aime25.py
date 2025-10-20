"""
注册 AIME 2025 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry


def load_aime25(path: str) -> List[Dict]:
    """
    加载 AIME 2025 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 AIME 2025 数据")
    return data


def aime25_prompt(item: Dict) -> str:
    """
    AIME 2025 prompt template
    """
    question = (
        item.get("question")
        or item.get("problem")
        or item.get("Problem")
        or ""
    )
    return (
        "You are a very talented mathematician who excels at AIME-style problems.\n"
        "Reason through the problem carefully before concluding with the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Answer: <three-digit AIME answer>\n\n"
        f"Problem:\n{question}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="aime25",
    loader=load_aime25,
    prompt_template=aime25_prompt,
    description="AIME 2025 mathematics problems (free-form solutions)",
    default_batch_size=16,
    default_max_length=512,
)
