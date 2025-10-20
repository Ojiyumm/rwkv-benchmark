"""
注册 RULER 长上下文评测数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry


def load_ruler(path: str) -> List[Dict]:
    """
    加载 RULER 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    for item in data:
        if not item.get("answer"):
            outputs = item.get("outputs") or item.get("answers")
            if isinstance(outputs, list) and outputs:
                item["answer"] = "\n".join(str(x) for x in outputs)
            elif isinstance(outputs, str):
                item["answer"] = outputs
    print(f"  ✓ 加载了 {len(data)} 条 RULER 数据")
    return data


def ruler_prompt(item: Dict) -> str:
    """
    RULER prompt template
    """
    base_prompt = (item.get("prompt") or item.get("input") or "").rstrip()
    gen_prefix = item.get("gen_prefix")
    
    parts = [base_prompt]
    if gen_prefix:
        parts.append(gen_prefix.rstrip())
    
    guidance = (
        "\nFirst reflect inside <think>...</think> using the provided context, "
        "then produce the final answer that satisfies the instructions.\n<think>"
    )
    parts.append(guidance)
    
    return "\n".join(part for part in parts if part)


DatasetRegistry.register(
    name="ruler",
    loader=load_ruler,
    prompt_template=ruler_prompt,
    description="RULER long-context benchmarks (JSON-prepared prompts)",
    default_batch_size=1,
    default_max_length=2048,
)
