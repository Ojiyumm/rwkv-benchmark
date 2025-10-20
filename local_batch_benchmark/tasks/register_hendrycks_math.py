"""
注册 Hendrycks Math 数据集
"""

import re
from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"Final Answer[:\s]*([^\n]+)", re.IGNORECASE)


def load_hendrycks_math(path: str) -> List[Dict]:
    """
    加载 Hendrycks Math 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    for item in data:
        if not item.get("answer"):
            solution = item.get("solution") or ""
            match = _BOXED_RE.search(solution)
            if match:
                item["answer"] = match.group(1).strip()
            else:
                match = _FINAL_RE.search(solution)
                if match:
                    item["answer"] = match.group(1).strip()
                else:
                    item["answer"] = ""
    print(f"  ✓ 加载了 {len(data)} 条 Hendrycks Math 数据")
    return data


def hendrycks_math_prompt(item: Dict) -> str:
    """
    Hendrycks Math prompt template
    """
    problem = item.get("problem") or item.get("question") or ""
    return (
        "You are a very talented competition mathematician.\n"
        "Solve the following problem with careful, step-by-step reasoning before presenting the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Answer: <final result>\n\n"
        f"Problem:\n{problem}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="hendrycks_math",
    loader=load_hendrycks_math,
    prompt_template=hendrycks_math_prompt,
    description="Hendrycks Math competition-level problems",
    default_batch_size=8,
    default_max_length=1024,
)
