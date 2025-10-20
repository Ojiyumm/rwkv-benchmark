"""
注册 GPQA Main (Zero-shot) 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def load_gpqa_main_zeroshot(path: str) -> List[Dict]:
    """
    加载 GPQA Main Zero-shot 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 GPQA Main 数据")
    return data


def _collect_options(item: Dict) -> List[str]:
    options = item.get("options")
    if isinstance(options, list) and options:
        return options
    collected = []
    for idx, label in enumerate(_OPTION_LABELS, start=1):
        value = item.get(f"choice{idx}")
        if value is None:
            value = item.get(label)  # 兼容 A/B/C/D 字段
        if value is None:
            break
        collected.append(str(value))
    return collected


def gpqa_main_prompt(item: Dict) -> str:
    """
    GPQA prompt template（四选一）
    """
    question = item.get("question") or item.get("Question") or ""
    options = _collect_options(item)
    option_lines = [
        f"{_OPTION_LABELS[idx]}. {opt}"
        for idx, opt in enumerate(options)
        if idx < len(_OPTION_LABELS)
    ]
    option_text = "\n".join(option_lines)
    subject = item.get("subject") or item.get("category") or "advanced science"
    
    return (
        f"You are a very talented expert in {subject}.\n"
        "Evaluate the following graduate-level multiple-choice question and select the single best answer.\n"
        "Respond using this format:\n"
        "<think>your rigorous reasoning</think>\n"
        "Final Answer: <option letter>\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"{option_text}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="gpqa_main_zeroshot",
    loader=load_gpqa_main_zeroshot,
    prompt_template=gpqa_main_prompt,
    description="GPQA main split in zero-shot format (multiple-choice)",
    default_batch_size=32,
    default_max_length=16,
)
