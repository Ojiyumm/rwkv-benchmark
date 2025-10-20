"""
注册 MMLU Pro Plus 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_OPTION_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


def load_mmlu_pro_plus(path: str) -> List[Dict]:
    """
    加载 MMLU Pro Plus 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 MMLU Pro Plus 数据")
    return data


def _collect_options(item: Dict) -> List[str]:
    options = item.get("options") or item.get("choices")
    if isinstance(options, list) and options:
        return [str(opt) for opt in options]
    collected = []
    for label in _OPTION_LABELS:
        value = item.get(label)
        if value is None:
            break
        collected.append(str(value))
    return collected


def mmlu_pro_plus_prompt(item: Dict) -> str:
    """
    MMLU Pro Plus prompt template
    """
    category = item.get("category") or item.get("subject")
    question = item.get("question", "")
    options = _collect_options(item)
    option_text = "\n".join(
        f"{_OPTION_LABELS[idx]}. {opt}"
        for idx, opt in enumerate(options)
        if idx < len(_OPTION_LABELS)
    )
    header = f"Category: {category}\n\n" if category else ""
    return f"{header}Question: {question}\n{option_text}\n\nAnswer:"


DatasetRegistry.register(
    name="mmlu_pro_plus",
    loader=load_mmlu_pro_plus,
    prompt_template=mmlu_pro_plus_prompt,
    description="MMLU Pro Plus (up to 16 options, professional subjects)",
    default_batch_size=32,
    default_max_length=32,
)

