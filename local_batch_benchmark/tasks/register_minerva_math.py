"""
注册 Minerva Math 数据集
"""

import re
from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"Final Answer[:\s]*([^\n]+)", re.IGNORECASE)


def load_minerva_math(path: str) -> List[Dict]:
    """
    加载 Minerva Math 数据集
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
    print(f"  ✓ 加载了 {len(data)} 条 Minerva Math 数据")
    return data


def minerva_math_prompt(item: Dict) -> str:
    """
    Minerva Math prompt template
    """
    problem = item.get("problem") or item.get("question") or ""
    return (
        "You are a very talented mathematician specializing in advanced competition problems.\n"
        "Work through the problem meticulously with step-by-step reasoning before presenting the final answer.\n"
        "Respond using this format:\n"
        "<think>your detailed reasoning</think>\n"
        "Answer: <final result>\n\n"
        f"Problem:\n{problem}\n\n"
        "Begin your reasoning now.\n<think>"
    )


# DatasetRegistry.register(
#     name="minerva_math",
#     loader=load_minerva_math,
#     prompt_template=minerva_math_prompt,
#     description="Minerva-style math problems with detailed solutions",
#     default_batch_size=4,
#     default_max_length=1536,
# )
