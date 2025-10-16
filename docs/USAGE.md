# 评估管线使用指南

## 🚀 快速开始

### 1. 最简单的方式 - 运行示例

```bash
# 查看可用的数据集和评估器
python run_pipeline.py --list

# 运行示例1: LAMBADA评估
python run_pipeline.py --example 1

# 运行示例2: 多任务评估
python run_pipeline.py --example 2
```

### 2. 命令行方式

```bash
# 完整的LAMBADA评估
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /home/rwkv/Peter/Albatross/eval/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256

# 快速测试（只跑100个样本）
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /home/rwkv/Peter/Albatross/eval/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256 \
    --limit 100

# 指定模型路径
python run_pipeline.py \
    --model-path /path/to/your/model \
    --dataset lambada \
    --dataset-path /path/to/lambada.jsonl \
    --evaluator perplexity
```

### 3. Python 代码方式

#### 方式A: 最简单 - quick_eval

```python
from pipeline import quick_eval

results = quick_eval(
    model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
    dataset_name="lambada",
    dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
    evaluator_name="perplexity",
    batch_size=256
)

print(f"Perplexity: {results['metrics']['perplexity']:.2f}")
print(f"Accuracy: {results['metrics']['accuracy']*100:.1f}%")
```

#### 方式B: 链式调用 - PipelineBuilder

```python
from pipeline import PipelineBuilder

results = (PipelineBuilder(
    model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
    output_dir="./eval_results"
)
.add_task(
    dataset_name="lambada",
    dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
    evaluator_name="perplexity",
    batch_size=256
)
.add_task(
    dataset_name="qa",
    dataset_path="/path/to/qa.jsonl",
    evaluator_name="generation",
    max_length=100
)
.run())

print(f"Completed {len(results)} tasks")
```

#### 方式C: 完全控制 - EvaluationPipeline

```python
from pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(
    model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
    output_dir="./eval_results",
    seed=42
)

# 任务1
results1 = pipeline.run(
    dataset_name="lambada",
    dataset_path="/home/rwkv/Peter/Albatross/eval/lambada_test.jsonl",
    evaluator_name="perplexity",
    batch_size=256,
    limit=100  # 快速测试
)

# 任务2
results2 = pipeline.run(
    dataset_name="qa",
    dataset_path="/path/to/qa.jsonl",
    evaluator_name="exact_match",
    batch_size=128
)
```

---

## 📦 模块说明

### 1. Dataset Registry（数据集注册器）

**作用**: 管理数据集加载和 prompt template

**内置数据集**:
- `lambada` - LAMBADA语言建模
- `qa` - 通用问答
- `completion` - 文本补全
- `cot` - Chain-of-Thought推理
- `math` - 数学问题

**注册自定义数据集**:

```python
from dataset_registry import DatasetRegistry
import json

# 定义加载函数
def load_my_data(path: str):
    with open(path, 'r') as f:
        return json.load(f)

# 定义 prompt template
def my_prompt_template(item: dict) -> str:
    return f"Question: {item['question']}\nAnswer:"

# 注册
DatasetRegistry.register(
    name='my_dataset',
    loader=load_my_data,
    prompt_template=my_prompt_template,
    description='My custom dataset',
    default_batch_size=128,
    default_max_length=100
)

# 使用
results = quick_eval(
    model_path="/path/to/model",
    dataset_name="my_dataset",  # 使用你注册的名字
    dataset_path="/path/to/data.json",
    evaluator_name="generation"
)
```

### 2. Evaluator Registry（评估器注册器）

**作用**: 管理不同的评估方法和指标计算

**内置评估器**:
- `perplexity` - 困惑度和准确率（适用于语言建模）
- `exact_match` - 精确匹配（适用于分类、简单QA）
- `generation` - 文本生成统计（tokens、throughput等）

**注册自定义评估器**:

```python
from evaluator_registry import EvaluatorRegistry

def my_evaluator(engine, data, batch_size=128, **kwargs):
    # 你的评估逻辑
    correct = 0
    total = len(data)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 生成
        tokens, _ = engine.generate_batch(prompts, max_length=100)
        predictions = engine.decode_tokens(tokens)
        
        # 评估逻辑
        for pred, item in zip(predictions, batch):
            if pred.strip() == item['reference'].strip():
                correct += 1
    
    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }

# 注册
EvaluatorRegistry.register(
    name='my_evaluator',
    evaluator=my_evaluator,
    description='My custom evaluator',
    metrics=['accuracy', 'correct', 'total']
)

# 使用
results = quick_eval(
    model_path="/path/to/model",
    dataset_name="lambada",
    dataset_path="/path/to/lambada.jsonl",
    evaluator_name="my_evaluator"  # 使用你注册的名字
)
```

---

## 🎯 实际使用案例

### 案例1: LAMBADA 完整评估

```bash
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /home/rwkv/Peter/Albatross/eval/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256
```

输出结果会保存在 `./eval_results/lambada_perplexity_TIMESTAMP.json`

### 案例2: 多数据集批量评估

创建一个脚本 `my_eval.py`:

```python
from pipeline import PipelineBuilder

model_path = "/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"

results = (PipelineBuilder(model_path, output_dir="./my_results")
    .add_task("lambada", "/path/to/lambada.jsonl", "perplexity", batch_size=256)
    .add_task("qa", "/path/to/qa.jsonl", "exact_match", batch_size=128)
    .add_task("math", "/path/to/math.json", "generation", batch_size=64, max_length=256)
    .run())

# 汇总结果会保存在 ./my_results/summary_TIMESTAMP.json
```

### 案例3: 添加新的数学评估数据集

```python
from dataset_registry import DatasetRegistry
from pipeline import quick_eval
import json

# 1. 注册数据集
def load_aime(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def aime_prompt(item: dict) -> str:
    return f"Problem: {item['problem']}\n\nSolution:\n"

DatasetRegistry.register(
    name='aime',
    loader=load_aime,
    prompt_template=aime_prompt,
    description='AIME math problems',
    default_batch_size=64,
    default_max_length=512
)

# 2. 运行评估
results = quick_eval(
    model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096",
    dataset_name="aime",
    dataset_path="/home/rwkv/Peter/makedata/data/aime_problems.json",
    evaluator_name="generation",
    limit=50  # 先测试50个
)
```

---

## 📁 输出文件说明

### 单任务结果文件
`./eval_results/DATASET_EVALUATOR_TIMESTAMP.json`

```json
{
  "timestamp": "20241016_123456",
  "dataset": {
    "name": "lambada",
    "path": "/path/to/lambada.jsonl",
    "size": 5153,
    "limit": null
  },
  "evaluator": "perplexity",
  "config": {
    "batch_size": 256,
    "max_length": 1
  },
  "metrics": {
    "perplexity": 15.23,
    "accuracy": 0.45,
    "total": 5153
  }
}
```

### 多任务汇总文件
`./eval_results/summary_TIMESTAMP.json`

包含所有任务的结果列表

---

## 💡 高级技巧

### 1. 快速调试（限制样本数）

```bash
python run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada.jsonl \
    --evaluator perplexity \
    --limit 100  # 只跑100个样本
```

### 2. 保存生成的文本

```python
results = quick_eval(
    model_path="/path/to/model",
    dataset_name="qa",
    dataset_path="/path/to/qa.jsonl",
    evaluator_name="generation",
    save_outputs=True,  # 保存生成的文本
    output_file="qa_outputs.jsonl"
)
```

### 3. 使用环境变量

```bash
export MODEL_PATH="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"

python run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada.jsonl \
    --evaluator perplexity
# 会自动使用 MODEL_PATH 环境变量
```

---

## 🔧 故障排除

### 问题1: 数据集未注册
```
ValueError: Dataset 'xxx' not registered
```
**解决**: 运行 `python run_pipeline.py --list` 查看可用数据集

### 问题2: 路径不存在
```
FileNotFoundError: [Errno 2] No such file or directory
```
**解决**: 检查 `--dataset-path` 和 `--model-path` 是否正确

### 问题3: 内存不足
**解决**: 减小 `--batch-size` 或使用 `--limit` 限制样本数

