# RWKV 本地批量评估系统

一个模块化的本地评估管线，用于高效运行 RWKV 模型的批量推理和评估任务。

## 📑 目录

- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [评估流程](#-评估流程)
- [使用方法](#-使用方法)
- [添加新数据集](#-添加新数据集)
- [添加新评估器](#-添加新评估器)
- [内置组件](#-内置组件)
- [Python API](#-python-api)
- [常见问题](#-常见问题)

---

## 🚀 快速开始

### 1. 列出所有可用组件

```bash
cd /home/rwkv/Peter/rwkveval/local_batch_benchmark
python3 run_pipeline.py --list
```

输出示例：
```
Registered Datasets:
  - lambada (LAMBADA language modeling)
  - mmlu_pro (MMLU Pro - Multi-task Language Understanding)
  - mmlu_pro_cot (MMLU Pro with Chain-of-Thought)
  ...

Registered Evaluators:
  - perplexity (Perplexity and accuracy evaluation)
  - mmlu_pro (MMLU Pro evaluator with subject-wise statistics)
  ...
```

### 2. 运行 MMLU-Pro 评估（推荐）

```bash
# 快速测试（100个样本）
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64 \
    --limit 100

# 完整评估（12,032个样本）
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64

# 保存推理结果到文件
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64 \
    --inferoutput ./results/mmlu_pro_inference.jsonl
```

### 3. 运行 LAMBADA 评估

```bash
python3 run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256 \
    --limit 100
```

---

## 📁 项目结构

```
local_batch_benchmark/
│
├── tasks/                          # 数据集注册模块（原 datasets/）
│   ├── __init__.py                 # 自动导入所有数据集
│   ├── dataset_registry.py         # 数据集注册器核心
│   └── register_mmlu_pro.py        # MMLU Pro 数据集注册示例
│
├── evaluators/                     # 评估器注册模块
│   ├── __init__.py                 # 自动导入所有评估器
│   ├── evaluator_registry.py       # 评估器注册器核心
│   └── register_mmlu_pro.py        # MMLU Pro 评估器注册示例
│
├── batch_engine.py                 # RWKV 批量推理引擎
├── pipeline.py                     # 评估管线核心
├── run_pipeline.py                 # 通用命令行入口
├── run_mmlu_pro.py                 # MMLU Pro 专用运行脚本
├── example_usage.py                # 使用示例（Python API）
└── README.md                       # 本文件
```

### 核心模块说明

#### 1. `batch_engine.py` - 推理引擎
- **功能**：加载 RWKV 模型，执行批量推理
- **主要类**：`RWKVInferenceEngine`
- **核心方法**：
  - `generate_batch()`: 批量生成文本
  - `decode_tokens()`: 解码 token 为文本
  - `generate_with_logprobs()`: 生成并返回 log 概率

#### 2. `tasks/` - 数据集模块
- **功能**：管理数据集加载和 prompt 模板
- **注册模式**：使用 `DatasetRegistry` 注册数据集
- **职责**：
  - 从文件加载原始数据
  - 应用 prompt 模板格式化数据
  - 提供默认批处理参数

#### 3. `evaluators/` - 评估器模块
- **功能**：执行具体的评估逻辑
- **注册模式**：使用 `EvaluatorRegistry` 注册评估器
- **职责**：
  - 批量推理
  - 计算评估指标（准确率、困惑度等）
  - 返回结构化结果

#### 4. `pipeline.py` - 管线核心
- **功能**：串联数据集、推理引擎、评估器
- **主要类**：
  - `EvaluationPipeline`: 完整评估管线
  - `PipelineBuilder`: 链式构建多任务评估
- **核心函数**：
  - `quick_eval()`: 快速运行单个评估

---

## 🔄 评估流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        评估完整流程                              │
└─────────────────────────────────────────────────────────────────┘

1. 初始化阶段
   ┌─────────────────────────────────────┐
   │  加载 RWKV 模型                      │
   │  └─ RWKVInferenceEngine.__init__()  │
   └─────────────────────────────────────┘
          ↓
2. 数据加载阶段
   ┌─────────────────────────────────────┐
   │  DatasetRegistry.load_dataset()     │
   │  ├─ 调用注册的 loader 函数          │
   │  ├─ 应用 prompt_template            │
   │  └─ 返回格式化的数据                │
   └─────────────────────────────────────┘
          ↓
3. 批量推理阶段
   ┌─────────────────────────────────────┐
   │  EvaluatorRegistry.evaluate()       │
   │  ├─ 分批处理数据 (batch_size)       │
   │  ├─ engine.generate_batch()         │
   │  ├─ engine.decode_tokens()          │
   │  └─ 收集所有预测结果                │
   └─────────────────────────────────────┘
          ↓
4. 评估计算阶段
   ┌─────────────────────────────────────┐
   │  评估器计算指标                      │
   │  ├─ 比较预测与参考答案              │
   │  ├─ 计算准确率/困惑度等指标         │
   │  └─ 生成详细统计信息                │
   └─────────────────────────────────────┘
          ↓
5. 结果保存阶段
   ┌─────────────────────────────────────┐
   │  保存结果                            │
   │  ├─ 评估指标 → JSON 文件            │
   │  ├─ 推理详情 → JSONL 文件 (可选)    │
   │  └─ 打印摘要到终端                  │
   └─────────────────────────────────────┘
```

---

## 📖 使用方法

### 方式 1: 使用通用命令行工具

```bash
python3 run_pipeline.py \
    --dataset <数据集名称> \
    --dataset-path <数据集路径> \
    --evaluator <评估器名称> \
    --model-path <模型路径> \
    --batch-size <批处理大小> \
    --max-length <最大生成长度> \
    --limit <样本数量限制> \
    --inferoutput <推理结果保存路径>
```

**参数说明：**

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--dataset` | ✅ | 数据集名称（已注册） | `mmlu_pro` |
| `--dataset-path` | ✅ | 数据集文件路径 | `/path/to/data` |
| `--evaluator` | ✅ | 评估器名称（已注册） | `mmlu_pro` |
| `--model-path` | ❌ | 模型路径 | `/path/to/model.pth` |
| `--batch-size` | ❌ | 批处理大小 | `64` |
| `--max-length` | ❌ | 最大生成长度 | `100` |
| `--limit` | ❌ | 限制样本数（用于测试） | `100` |
| `--inferoutput` | ❌ | 保存推理结果的路径 | `./results.jsonl` |
| `--seed` | ❌ | 随机种子 | `42` |

**示例：**

```bash
# 示例 1: LAMBADA 评估
python3 run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256 \
    --limit 500

# 示例 2: MMLU-Pro 评估（保存推理结果）
python3 run_pipeline.py \
    --dataset mmlu_pro \
    --dataset-path /path/to/mmlupro \
    --evaluator mmlu_pro \
    --batch-size 64 \
    --inferoutput ./results/mmlu_pro_predictions.jsonl
```

### 方式 2: 使用 MMLU-Pro 专用脚本

```bash
python3 run_mmlu_pro.py \
    --dataset-path <数据集路径> \
    --model-path <模型路径> \
    --batch-size <批处理大小> \
    --max-length <最大生成长度> \
    --cot \
    --limit <样本数量限制> \
    --inferoutput <推理结果保存路径>
```

**参数说明：**

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dataset-path` | ✅ | MMLU-Pro 数据集路径 | - |
| `--model-path` | ❌ | 模型路径 | `/home/rwkv/models/...` |
| `--batch-size` | ❌ | 批处理大小 | `64` |
| `--max-length` | ❌ | 最大生成长度 | `10` |
| `--cot` | ❌ | 使用 Chain-of-Thought | `False` |
| `--limit` | ❌ | 限制样本数 | 全部 |
| `--inferoutput` | ❌ | 保存推理结果 | 不保存 |

**示例：**

```bash
# 标准 MMLU-Pro 评估
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64

# 使用 Chain-of-Thought
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 32 \
    --max-length 200 \
    --cot

# 快速测试 + 保存结果
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64 \
    --limit 100 \
    --inferoutput ./debug_results.jsonl
```

### 方式 3: 使用 Python API

**简单评估：**

```python
from pipeline import quick_eval

results = quick_eval(
    model_path="/path/to/model",
    dataset_name="mmlu_pro",
    dataset_path="/path/to/mmlupro",
    evaluator_name="mmlu_pro",
    batch_size=64,
    limit=100,
    inferoutput="./results.jsonl"  # 可选
)

print(f"Accuracy: {results['metrics']['accuracy']*100:.2f}%")
```

**链式构建多任务：**

```python
from pipeline import PipelineBuilder

builder = PipelineBuilder(
    model_path="/path/to/model",
    output_dir="./eval_results"
)

builder.add_task(
    dataset_name="lambada",
    dataset_path="/path/to/lambada.jsonl",
    evaluator_name="perplexity",
    batch_size=256
).add_task(
    dataset_name="mmlu_pro",
    dataset_path="/path/to/mmlupro",
    evaluator_name="mmlu_pro",
    batch_size=64
).run()
```

**完整自定义：**

```python
from pipeline import EvaluationPipeline
from tasks import DatasetRegistry
from evaluators import EvaluatorRegistry

# 初始化管线
pipeline = EvaluationPipeline(
    model_path="/path/to/model",
    output_dir="./eval_results",
    seed=42
)

# 运行评估
results = pipeline.run(
    dataset_name="mmlu_pro",
    dataset_path="/path/to/mmlupro",
    evaluator_name="mmlu_pro",
    batch_size=64,
    max_length=10,
    limit=None,
    inferoutput="./mmlu_results.jsonl"  # 保存推理结果
)

# 获取指标
print(f"Overall Accuracy: {results['metrics']['accuracy']:.2%}")
print(f"Correct: {results['metrics']['correct']}/{results['metrics']['total']}")

# 按学科查看
for subject, acc in results['metrics']['subject_accuracies'].items():
    print(f"{subject}: {acc:.2%}")
```

---

## ➕ 添加新数据集

### 步骤 1: 创建注册文件

在 `tasks/` 目录下创建 `register_your_dataset.py`：

```python
"""
注册你的数据集
"""

import json
from typing import List, Dict
from .dataset_registry import DatasetRegistry


# ==================== 数据加载器 ====================

def load_your_dataset(path: str) -> List[Dict]:
    """
    从文件加载数据
    
    返回格式：List[Dict]，每个 Dict 包含原始数据的所有字段
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  ✓ 加载了 {len(data)} 条数据")
    return data


# ==================== Prompt Template ====================

def your_prompt_template(item: Dict) -> str:
    """
    将原始数据转换为 prompt
    
    Args:
        item: 原始数据的一条记录
        
    Returns:
        格式化后的 prompt 字符串
    """
    question = item.get('question', '')
    context = item.get('context', '')
    
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"
    
    return prompt


# ==================== 注册 ====================

DatasetRegistry.register(
    name='your_dataset',
    loader=load_your_dataset,
    prompt_template=your_prompt_template,
    description='Your dataset description',
    default_batch_size=128,
    default_max_length=100
)
```

### 步骤 2: 在 `__init__.py` 中导入

编辑 `tasks/__init__.py`，添加：

```python
try:
    from . import register_your_dataset
except ImportError as e:
    print(f"Warning: Could not import register_your_dataset: {e}")
```

### 步骤 3: 使用新数据集

```bash
python3 run_pipeline.py \
    --dataset your_dataset \
    --dataset-path /path/to/your_data.json \
    --evaluator exact_match
```

---

## ➕ 添加新评估器

### 步骤 1: 创建注册文件

在 `evaluators/` 目录下创建 `register_your_evaluator.py`：

```python
"""
注册你的评估器
"""

from typing import List, Dict
from tqdm import tqdm
from .evaluator_registry import EvaluatorRegistry


def your_evaluator(
    engine,
    data: List[Dict],
    batch_size: int = 128,
    max_length: int = 100,
    **kwargs
) -> Dict[str, float]:
    """
    你的评估器
    
    Args:
        engine: RWKVInferenceEngine 实例
        data: 数据列表，每项包含 'prompt' 和 'reference'
        batch_size: 批处理大小
        max_length: 最大生成长度
        **kwargs: 其他参数（如 inferoutput）
        
    Returns:
        评估指标字典
    """
    print(f"\n=== Running Your Evaluator ===")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {batch_size}\n")
    
    correct = 0
    total = 0
    
    # 使用 tqdm 显示进度
    pbar = tqdm(total=len(data), desc="Evaluating", unit="samples")
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 批量推理
        tokens, _ = engine.generate_batch(prompts, max_length=max_length)
        predictions = engine.decode_tokens(tokens)
        
        # 评估每个样本
        for pred, item in zip(predictions, batch):
            reference = item['reference']
            
            # 你的评估逻辑
            if pred.strip().lower() == reference.strip().lower():
                correct += 1
            total += 1
        
        # 更新进度条
        pbar.update(len(batch))
        pbar.set_postfix({'accuracy': f'{correct/total*100:.2f}%'})
    
    pbar.close()
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    # 打印结果
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    return results


# ==================== 注册 ====================

EvaluatorRegistry.register(
    name='your_evaluator',
    evaluator=your_evaluator,
    description='Your evaluator description',
    metrics=['accuracy', 'correct', 'total']
)
```

### 步骤 2: 在 `__init__.py` 中导入

编辑 `evaluators/__init__.py`，添加：

```python
try:
    from . import register_your_evaluator
except ImportError as e:
    print(f"Warning: Could not import register_your_evaluator: {e}")
```

### 步骤 3: 使用新评估器

```bash
python3 run_pipeline.py \
    --dataset your_dataset \
    --dataset-path /path/to/data.json \
    --evaluator your_evaluator
```

---

## 📦 内置组件

### 内置数据集 (tasks/)

| 名称 | 描述 | 默认 batch_size | 默认 max_length |
|------|------|----------------|----------------|
| `lambada` | LAMBADA 语言建模 | 256 | 1 |
| `qa` | 通用问答 | 128 | 100 |
| `completion` | 文本补全 | 128 | 100 |
| `cot` | Chain-of-Thought 推理 | 64 | 200 |
| `math` | 数学问题 | 64 | 150 |
| `mmlu_pro` | MMLU Pro 多选题 | 64 | 10 |
| `mmlu_pro_cot` | MMLU Pro (CoT版本) | 32 | 200 |

### 内置评估器 (evaluators/)

| 名称 | 描述 | 支持的指标 |
|------|------|-----------|
| `perplexity` | 困惑度和准确率 | `perplexity`, `accuracy`, `correct`, `total` |
| `exact_match` | 精确匹配 | `exact_match`, `correct`, `total` |
| `generation` | 生成统计 | `avg_length`, `total_tokens` |
| `mmlu_pro` | MMLU Pro 多选题评估 | `accuracy`, `correct`, `total`, `subject_accuracies`, `num_subjects` |

---

## 💻 Python API

### DatasetRegistry API

```python
from tasks import DatasetRegistry

# 列出所有数据集
datasets = DatasetRegistry.list_datasets()
print(datasets)  # ['lambada', 'mmlu_pro', ...]

# 获取数据集配置
config = DatasetRegistry.get('mmlu_pro')
print(config.description)
print(config.default_batch_size)

# 加载数据集
data = DatasetRegistry.load_dataset(
    name='mmlu_pro',
    path='/path/to/mmlupro',
    limit=100  # 可选：限制样本数
)

# 每条数据包含：
# - 'prompt': 格式化后的 prompt
# - 'reference': 参考答案
# - 'raw': 原始数据
for item in data[:3]:
    print(item['prompt'])
    print(item['reference'])
```

### EvaluatorRegistry API

```python
from evaluators import EvaluatorRegistry

# 列出所有评估器
evaluators = EvaluatorRegistry.list_evaluators()
print(evaluators)  # ['perplexity', 'mmlu_pro', ...]

# 获取评估器配置
config = EvaluatorRegistry.get('mmlu_pro')
print(config.description)
print(config.metrics)

# 运行评估
from batch_engine import RWKVInferenceEngine

engine = RWKVInferenceEngine(model_path="/path/to/model")
data = [...]  # 加载的数据

results = EvaluatorRegistry.evaluate(
    name='mmlu_pro',
    engine=engine,
    data=data,
    batch_size=64,
    inferoutput='./results.jsonl'  # 可选
)

print(results)
# {'accuracy': 0.35, 'correct': 350, 'total': 1000, ...}
```

### RWKVInferenceEngine API

```python
from batch_engine import RWKVInferenceEngine

# 初始化引擎
engine = RWKVInferenceEngine(
    model_path="/path/to/model.pth",
    vocab_path=None,  # 可选，默认使用标准词汇表
    vocab_size=65536,
    head_size=64,
    seed=42
)

# 批量生成
prompts = ["Question 1", "Question 2", "Question 3"]
tokens, inference_time = engine.generate_batch(
    prompts=prompts,
    max_length=100,
    noise=0.0
)

# 解码
texts = engine.decode_tokens(tokens)
for text in texts:
    print(text)

# 生成并返回 log 概率（用于 logprobs 任务）
logprobs_result = engine.generate_with_logprobs(
    prompt="Question",
    max_length=10,
    top_logprobs=5,
    echo=True
)
```

---

## 📊 输出格式

### 1. 评估指标文件 (JSON)

保存在 `./eval_results/` 目录：

```json
{
  "timestamp": "2024-10-16 12:34:56",
  "model_path": "/path/to/model",
  "dataset": "mmlu_pro",
  "evaluator": "mmlu_pro",
  "config": {
    "batch_size": 64,
    "max_length": 10,
    "seed": 42
  },
  "metrics": {
    "accuracy": 0.3542,
    "correct": 4263,
    "total": 12032,
    "num_subjects": 14,
    "subject_accuracies": {
      "Math": 0.42,
      "Physics": 0.38,
      "Biology": 0.35,
      ...
    }
  },
  "inference_time": 1234.56
}
```

### 2. 推理结果文件 (JSONL)

如果指定了 `--inferoutput`，保存每条样本的推理详情：

```jsonl
{"question": "What is...", "options": ["A...", "B...", ...], "category": "Math", "prompt": "Question: ...", "prediction": "A", "predicted_answer": "A", "correct_answer": "B", "is_correct": false}
{"question": "Which...", "options": ["A...", "B...", ...], "category": "Physics", "prompt": "Question: ...", "prediction": "C", "predicted_answer": "C", "correct_answer": "C", "is_correct": true}
...
```

每行一个 JSON 对象，包含：
- `question`: 原始问题
- `options`: 选项列表
- `category`: 学科/类别
- `prompt`: 完整的输入 prompt
- `prediction`: 模型的原始输出
- `predicted_answer`: 提取的答案
- `correct_answer`: 正确答案
- `is_correct`: 是否正确

---

## ❓ 常见问题

### Q1: 如何调整批处理大小以适应显存？

A: 根据你的 GPU 显存调整 `--batch-size`：

```bash
# 显存充足 (24GB+)
--batch-size 128

# 显存中等 (12GB)
--batch-size 64

# 显存较小 (8GB)
--batch-size 32
```

### Q2: 如何只测试部分数据进行快速验证？

A: 使用 `--limit` 参数：

```bash
python3 run_pipeline.py \
    --dataset mmlu_pro \
    --dataset-path /path/to/data \
    --evaluator mmlu_pro \
    --limit 100  # 只测试前100个样本
```

### Q3: 推理结果保存在哪里？

A: 
- **评估指标**: `./eval_results/<dataset>_<evaluator>_<timestamp>.json`
- **推理详情** (如果指定 `--inferoutput`): 你指定的路径

### Q4: 如何查看实时进度？

A: 评估过程会显示 tqdm 进度条：

```
Evaluating MMLU-Pro:  45%|████████▌      | 5400/12032 [02:15<02:45, 40.1samples/s, accuracy=28.45%]
```

### Q5: MMLU-Pro 使用哪种评估方法？

A: 当前使用**生成式方法**（官方标准）：
1. 模型生成输出（通常是一个字母 A-J）
2. 从输出中提取答案字母
3. 与正确答案比较

这与 MMLU-Pro 官方评估方法一致。

### Q6: 如何使用 Chain-of-Thought 评估？

A: 对于 MMLU-Pro，使用 `--cot` 参数：

```bash
python3 run_mmlu_pro.py \
    --dataset-path /path/to/mmlupro \
    --cot \
    --max-length 200  # CoT 需要更长的生成长度
```

### Q7: 数据集和 API 评估有什么区别？

| 特性 | 本地批量评估 | API 评估 |
|------|------------|---------|
| 位置 | `local_batch_benchmark/` | `api_benchmark/` |
| 运行方式 | 直接调用推理引擎 | 通过 HTTP API |
| 适用场景 | 本地测试、快速迭代 | 与 lm_eval 等外部工具集成 |
| 性能 | 更快（无网络开销） | 稍慢（有 HTTP 开销） |
| 灵活性 | 完全控制 | 标准化接口 |

### Q8: 如何添加对新数据集的支持？

A: 参见 [添加新数据集](#-添加新数据集) 章节。

### Q9: 评估很慢怎么办？

A: 优化建议：
1. 增加 `--batch-size`（如果显存允许）
2. 减少 `--max-length`（如果不需要长输出）
3. 使用 `--limit` 快速测试
4. 确保使用 GPU 推理

### Q10: 如何自定义 prompt 模板？

A: 修改对应数据集的 `register_*.py` 文件中的 prompt 模板函数，或创建新的数据集注册。

---

## 📞 支持

如有问题，请查看：
- **示例代码**: `example_usage.py`
- **测试脚本**: `run_pipeline.py --example 1`
- **内联文档**: 各模块的 docstring

---

## 📝 许可

本项目遵循 Apache 2.0 许可证。

---

**Happy Evaluating! 🚀**
