"""
注册 GSM8K 数据集
"""

import re
from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_GSM8K_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9.,]+)")


def load_gsm8k(path: str) -> List[Dict]:
    """
    加载 GSM8K 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    for item in data:
        answer = item.get("answer")
        if isinstance(answer, str):
            match = _GSM8K_ANSWER_RE.search(answer)
            if match:
                item["answer"] = match.group(1).replace(",", "")
            else:
                item["answer"] = answer.strip()
    print(f"  ✓ 加载了 {len(data)} 条 GSM8K 数据")
    return data


def gsm8k_prompt(item: Dict) -> str:
    """
    GSM8K prompt template
    """
    question = item.get("question", "")
    return (
        "You are a brilliant mathematics tutor who excels at solving grade-school word problems.\n"
        "Carefully reason through the problem step by step before giving the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Answer: <final numeric answer>\n\n"
        f"Problem:\n{question}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="gsm8k",
    loader=load_gsm8k,
    prompt_template=gsm8k_prompt,
    description="GSM8K grade-school math word problems",
    default_batch_size=16,
    default_max_length=256,
)
