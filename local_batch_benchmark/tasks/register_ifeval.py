"""
注册 IFEval 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry


def load_ifeval(path: str) -> List[Dict]:
    """
    加载 IFEval 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 IFEval 数据")
    return data


def ifeval_prompt(item: Dict) -> str:
    """
    IFEval prompt template
    """
    instruction = (item.get("prompt") or item.get("instruction") or "").strip()
    prefix = (
        "You are a highly capable assistant who follows instructions precisely.\n"
        "Before responding, reason carefully inside <think>...</think>, then provide the final reply that obeys every requirement.\n\n"
    )
    return (
        f"{prefix}"
        f"Instruction:\n{instruction}\n\n"
        "Begin.\n<think>"
    )


DatasetRegistry.register(
    name="ifeval",
    loader=load_ifeval,
    prompt_template=ifeval_prompt,
    description="Instruction Following Evaluation prompts",
    default_batch_size=8,
    default_max_length=512,
)
