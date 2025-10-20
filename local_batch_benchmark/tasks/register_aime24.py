"""
注册 AIME 2024 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry


def load_aime24(path: str) -> List[Dict]:
    """
    加载 AIME 2024 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 AIME 2024 数据")
    return data


def aime24_prompt(item: Dict) -> str:
    """
    AIME 2024 prompt template
    """
    question = (
        item.get("question")
        or item.get("problem")
        or item.get("Problem")
        or ""
    )
    return (
        "You are a very talented mathematician who excels at AIME-style problems.\n"
        "Work through the problem carefully with step-by-step reasoning before stating the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Answer: <three-digit AIME answer>\n\n"
        f"Problem:\n{question}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="aime24",
    loader=load_aime24,
    prompt_template=aime24_prompt,
    description="AIME 2024 mathematics problems (free-form solutions)",
    default_batch_size=16,
    default_max_length=512,
)
