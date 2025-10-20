"""
注册 MMLU 数据集
"""

from typing import Dict, List

from datasets import load_dataset

from .dataset_registry import DatasetRegistry

_OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def load_mmlu(path: str) -> List[Dict]:
    """
    加载 MMLU 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 MMLU 数据")
    return data


def _collect_options(item: Dict) -> List[str]:
    options = []
    for label in _OPTION_LABELS:
        value = item.get(label)
        if value is None:
            break
        options.append(str(value))
    return options


def mmlu_prompt(item: Dict) -> str:
    """
    MMLU prompt template
    """
    question = item.get("question", "")
    options = _collect_options(item)
    option_lines = [
        f"{label}. {opt}"
        for label, opt in zip(_OPTION_LABELS, options)
    ]
    option_text = "\n".join(option_lines)
    
    subject = item.get("subject") or "general knowledge"
    
    return (
        f"You are a very talented expert in {subject}.\n"
        "Carefully analyze the multiple-choice question and select the single best option.\n"
        "Respond using this format:\n"
        "<think>your step-by-step reasoning</think>\n"
        "Final Answer: <option letter>\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"{option_text}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="mmlu",
    loader=load_mmlu,
    prompt_template=mmlu_prompt,
    description="Massive Multitask Language Understanding (standard four-choice)",
    default_batch_size=64,
    default_max_length=16,
)
