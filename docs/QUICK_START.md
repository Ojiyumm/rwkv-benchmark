# 🚀 快速开始 - 评估管线

## 📦 已有数据集，直接运行

### LAMBADA 评估
```bash
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /home/rwkv/Peter/Albatross/eval/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256
```

### 查看可用组件
```bash
python run_pipeline.py --list
```

### 快速测试（限制样本数）
```bash
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada.jsonl \
    --evaluator perplexity \
    --limit 100  # 只跑100个样本
```

---

## ➕ 添加新数据集（以 MMLU Pro 为例）

### 第1步：创建注册文件 `register_mmlu_pro.py`

```python
from dataset_registry import DatasetRegistry
from evaluator_registry import EvaluatorRegistry
import json

# 1️⃣ 数据加载器
def load_mmlu_pro(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

# 2️⃣ Prompt Template
def mmlu_pro_prompt(item: dict) -> str:
    question = item['question']
    options = item['options']
    
    option_text = '\n'.join([
        f"{chr(65+i)}. {opt}" 
        for i, opt in enumerate(options)
    ])
    
    return f"Question: {question}\n\n{option_text}\n\nAnswer:"

# 3️⃣ 评估器（可选，可以用内置的）
def mmlu_pro_evaluator(engine, data, batch_size=64, **kwargs):
    correct = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        tokens, _ = engine.generate_batch(prompts, max_length=10)
        predictions = engine.decode_tokens(tokens)
        
        for pred, item in zip(predictions, batch):
            pred_answer = pred.strip().upper()[0] if pred else ''
            if pred_answer == item['reference']:
                correct += 1
    
    return {'accuracy': correct / len(data)}

# 4️⃣ 注册
DatasetRegistry.register(
    name='mmlu_pro',
    loader=load_mmlu_pro,
    prompt_template=mmlu_pro_prompt,
    description='MMLU Pro',
    default_batch_size=64,
    default_max_length=10
)

EvaluatorRegistry.register(
    name='mmlu_pro',
    evaluator=mmlu_pro_evaluator,
    description='Multi-choice accuracy',
    metrics=['accuracy']
)

print("✓ MMLU Pro registered!")
```

### 第2步：使用

**命令行：**
```bash
python run_pipeline.py \
    --dataset mmlu_pro \
    --dataset-path /path/to/mmlu_pro.jsonl \
    --evaluator mmlu_pro
```

**Python 代码：**
```python
from register_mmlu_pro import *  # 自动注册
from pipeline import quick_eval

results = quick_eval(
    model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
    dataset_name="mmlu_pro",
    dataset_path="/path/to/mmlu_pro.jsonl",
    evaluator_name="mmlu_pro"
)
```

---

## 🎨 常见 Prompt 模板

### 简单 QA
```python
def qa_prompt(item):
    return f"Q: {item['question']}\nA:"
```

### 多选题
```python
def mcq_prompt(item):
    opts = '\n'.join([f"{chr(65+i)}. {o}" for i, o in enumerate(item['options'])])
    return f"{item['question']}\n\n{opts}\n\nAnswer:"
```

### Chain-of-Thought
```python
def cot_prompt(item):
    return f"{item['question']}\n\nLet's think step by step:\n"
```

### 指令跟随
```python
def instruction_prompt(item):
    return f"{item['instruction']}\n\nInput: {item['input']}\nOutput:"
```

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `dataset_registry.py` | 数据集注册器（加载+prompt模板） |
| `evaluator_registry.py` | 评估器注册器（指标计算） |
| `pipeline.py` | 评估管线核心 |
| `run_pipeline.py` | 命令行入口 |
| `batch_engine.py` | RWKV 推理引擎 |
| `register_*.py` | 各数据集注册文件 |

---

## 🔧 内置组件

### 内置数据集
- `lambada` - LAMBADA 语言建模
- `qa` - 通用问答
- `completion` - 文本补全
- `cot` - Chain-of-Thought
- `math` - 数学问题

### 内置评估器
- `perplexity` - 困惑度 + 准确率（语言建模）
- `exact_match` - 精确匹配（分类、QA）
- `generation` - 生成统计（throughput、tokens）

---

## 📊 结果输出

评估结果保存在 `./eval_results/` 目录：

```
eval_results/
├── lambada_perplexity_20241016_123456.json  # 单任务结果
├── mmlu_pro_mmlu_pro_20241016_123500.json
└── summary_20241016_123600.json             # 多任务汇总
```

结果格式：
```json
{
  "timestamp": "20241016_123456",
  "dataset": {"name": "lambada", "size": 5153},
  "evaluator": "perplexity",
  "metrics": {
    "perplexity": 15.23,
    "accuracy": 0.45
  }
}
```

---

## 💡 实用技巧

### 1. 快速测试（限制样本）
```bash
--limit 100  # 只测试100个样本
```

### 2. 自定义输出目录
```bash
--output-dir ./my_results
```

### 3. 使用环境变量
```bash
export MODEL_PATH="/path/to/model"
python run_pipeline.py --dataset lambada --dataset-path /path/to/data.jsonl --evaluator perplexity
```

### 4. 批量运行多个任务
```python
from pipeline import PipelineBuilder

(PipelineBuilder("/path/to/model")
    .add_task("lambada", "/path/to/lambada.jsonl", "perplexity")
    .add_task("mmlu_pro", "/path/to/mmlu.jsonl", "mmlu_pro")
    .add_task("gsm8k", "/path/to/gsm8k.jsonl", "generation")
    .run())
```

---

## 📚 详细文档

- **使用指南**: `USAGE.md`
- **添加数据集**: `HOW_TO_ADD_NEW_DATASET.md`
- **MMLU Pro 示例**: `register_mmlu_pro.py`
- **完整文档**: `README.md`

