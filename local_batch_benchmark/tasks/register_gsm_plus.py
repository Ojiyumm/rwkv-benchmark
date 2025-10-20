"""
注册 GSM-Plus 数据集
"""

import re
from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_GSM_PLUS_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9.,]+)")


def load_gsm_plus(path: str) -> List[Dict]:
    """
    加载 GSM-Plus 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    for item in data:
        if not item.get("answer"):
            solution = item.get("solution") or item.get("final_answer")
            item["answer"] = _extract_answer(solution or "")
    print(f"  ✓ 加载了 {len(data)} 条 GSM-Plus 数据")
    return data


def _extract_answer(solution: str) -> str:
    if not solution:
        return ""
    match = _GSM_PLUS_ANSWER_RE.search(solution)
    if match:
        return match.group(1).replace(",", "")
    return solution.strip()


def gsm_plus_prompt(item: Dict) -> str:
    """
    GSM-Plus prompt template
    """
    question = item.get("question", "")
    return f"Question: {question}\n\nAnswer:"


def _register():
    DatasetRegistry.register(
        name="gsm_plus",
        loader=load_gsm_plus,
        prompt_template=gsm_plus_prompt,
        description="GSM-Plus math word problems with rationales",
        default_batch_size=16,
        default_max_length=256,
    )


_register()
