"""
注册 MMLU-ProX 数据集
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
]


def load_mmlu_prox(path: str) -> List[Dict]:
    """
    加载 MMLU-ProX 数据集
    """
    ds = load_dataset("json", data_files=path, split="train")
    data = list(ds)
    print(f"  ✓ 加载了 {len(data)} 条 MMLU-ProX 数据")
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


def mmlu_prox_prompt(item: Dict) -> str:
    """
    MMLU-ProX prompt template
    """
    language = item.get("language")
    subject = item.get("subject")
    question = item.get("question", "")
    options = _collect_options(item)
    option_text = "\n".join(
        f"{_OPTION_LABELS[idx]}. {opt}"
        for idx, opt in enumerate(options)
        if idx < len(_OPTION_LABELS)
    )

    domain = subject or "cross-disciplinary knowledge"
    language_hint = f" in {language}" if language else ""
    role_line = f"You are a very talented expert in {domain}{language_hint}."
    
    header_parts = []
    if language:
        header_parts.append(f"Language: {language}")
    if subject:
        header_parts.append(f"Subject: {subject}")
    header_block = f"{' | '.join(header_parts)}\n\n" if header_parts else ""

    return (
        f"{role_line}\n"
        "Carefully analyze the question and choose the single best option using the available evidence.\n"
        "Respond using this format:\n"
        "<think>your step-by-step reasoning</think>\n"
        "Final Answer: <option letter>\n\n"
        f"{header_block}"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        f"{option_text}\n\n"
        "Begin your reasoning now.\n<think>"
    )


DatasetRegistry.register(
    name="mmlu_prox",
    loader=load_mmlu_prox,
    prompt_template=mmlu_prox_prompt,
    description="MMLU-ProX cross-lingual professional subjects",
    default_batch_size=32,
    default_max_length=32,
)
