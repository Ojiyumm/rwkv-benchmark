# 如何添加新的数据集和评估器

## 📝 快速步骤

添加新的测评任务只需要 **4 步**：

1. **定义数据加载器** - 如何读取数据文件
2. **定义 Prompt Template** - 如何格式化输入
3. **定义评估器**（可选） - 如何计算指标
4. **注册** - 注册到系统中

## 🎯 完整示例：添加 MMLU Pro

### 步骤 1: 创建注册文件

创建 `register_mmlu_pro.py`：

```python
from dataset_registry import DatasetRegistry
from evaluator_registry import EvaluatorRegistry
import json

# ========== 1. 数据加载器 ==========
def load_mmlu_pro(path: str):
    """加载 MMLU Pro 数据"""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

# ========== 2. Prompt Template ==========
def mmlu_pro_prompt(item: dict) -> str:
    """格式化为多选题"""
    question = item['question']
    options = item['options']
    
    # 构建选项 A, B, C, D...
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    option_text = '\n'.join([
        f"{option_labels[i]}. {opt}" 
        for i, opt in enumerate(options)
    ])
    
    return f"Question: {question}\n\n{option_text}\n\nAnswer:"

# ========== 3. 评估器（可选） ==========
def mmlu_pro_evaluator(engine, data, batch_size=64, **kwargs):
    """多选题评估"""
    correct = 0
    total = len(data)
    
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 生成答案
        tokens, _ = engine.generate_batch(prompts, max_length=10)
        predictions = engine.decode_tokens(tokens)
        
        # 评估
        for pred, item in zip(predictions, batch):
            # 提取预测的字母
            pred_answer = None
            for char in pred.strip().upper():
                if char in option_labels:
                    pred_answer = char
                    break
            
            # 获取正确答案
            correct_answer = item['reference']
            if pred_answer == correct_answer:
                correct += 1
    
    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }

# ========== 4. 注册 ==========
def register_mmlu_pro():
    # 注册数据集
    DatasetRegistry.register(
        name='mmlu_pro',
        loader=load_mmlu_pro,
        prompt_template=mmlu_pro_prompt,
        description='MMLU Pro multi-choice questions',
        default_batch_size=64,
        default_max_length=10
    )
    
    # 注册评估器
    EvaluatorRegistry.register(
        name='mmlu_pro',
        evaluator=mmlu_pro_evaluator,
        description='Multi-choice accuracy',
        metrics=['accuracy', 'correct', 'total']
    )
    
    print("✓ MMLU Pro registered!")

# 导入时自动注册
register_mmlu_pro()
```

### 步骤 2: 使用

**方法 1: 命令行**

```bash
# 直接使用（会自动注册）
python run_pipeline.py \
    --dataset mmlu_pro \
    --dataset-path /path/to/mmlu_pro.jsonl \
    --evaluator mmlu_pro \
    --batch-size 64
```

**方法 2: Python 代码**

```python
from register_mmlu_pro import register_mmlu_pro  # 导入时自动注册
from pipeline import quick_eval

results = quick_eval(
    model_path="/path/to/model",
    dataset_name="mmlu_pro",
    dataset_path="/path/to/mmlu_pro.jsonl",
    evaluator_name="mmlu_pro"
)
```

**方法 3: 专用脚本**

```bash
# 使用专门为 MMLU Pro 创建的脚本
python run_mmlu_pro.py \
    --dataset-path /path/to/mmlu_pro.jsonl \
    --batch-size 64 \
    --limit 100  # 快速测试
```

---

## 🔧 详细说明

### 1. 数据加载器函数

数据加载器负责读取文件并返回数据列表：

```python
def load_your_dataset(path: str) -> List[Dict]:
    """
    输入: 文件路径
    输出: 数据列表，每个元素是一个字典
    """
    # 读取文件
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 或 [json.loads(line) for line in f]
    
    return data
```

**支持的数据格式：**
- JSON: `json.load(f)`
- JSONL: `[json.loads(line) for line in f]`
- CSV: 使用 `pandas.read_csv()`
- 其他格式: 自定义解析逻辑

### 2. Prompt Template 函数

Prompt Template 负责将数据格式化为模型输入：

```python
def your_prompt_template(item: Dict) -> str:
    """
    输入: 数据项（字典）
    输出: 格式化的 prompt 字符串
    """
    # 从 item 中提取字段
    question = item['question']
    context = item.get('context', '')  # 可选字段
    
    # 组合成 prompt
    if context:
        return f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"
```

**常见 Prompt 模式：**

```python
# 1. 简单问答
def qa_prompt(item):
    return f"Q: {item['question']}\nA:"

# 2. 指令跟随
def instruction_prompt(item):
    return f"{item['instruction']}\n\nInput: {item['input']}\nOutput:"

# 3. 多选题
def multiple_choice_prompt(item):
    options = '\n'.join([f"{i}. {opt}" for i, opt in enumerate(item['options'])])
    return f"{item['question']}\n\n{options}\n\nAnswer:"

# 4. Chain-of-Thought
def cot_prompt(item):
    return f"{item['question']}\n\nLet's think step by step:\n"

# 5. Few-shot
def fewshot_prompt(item):
    examples = "Q: 2+2\nA: 4\n\nQ: 3+3\nA: 6\n\n"
    return f"{examples}Q: {item['question']}\nA:"
```

### 3. 评估器函数（可选）

如果内置评估器不满足需求，可以自定义：

```python
def your_evaluator(engine, data, batch_size=128, **kwargs):
    """
    输入: 
      - engine: 推理引擎
      - data: 数据列表（已经应用了 prompt template）
      - batch_size: 批量大小
      - **kwargs: 其他参数
    
    输出: 
      - Dict[str, float]: 评估指标
    """
    correct = 0
    total = len(data)
    
    # 批量处理
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [item['prompt'] for item in batch]
        
        # 生成
        tokens, _ = engine.generate_batch(prompts, max_length=100)
        predictions = engine.decode_tokens(tokens)
        
        # 评估逻辑
        for pred, item in zip(predictions, batch):
            if your_matching_logic(pred, item['reference']):
                correct += 1
    
    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }
```

**内置评估器：**
- `perplexity` - 困惑度评估（语言建模）
- `exact_match` - 精确匹配（分类、简单QA）
- `generation` - 生成统计（throughput、token数等）

### 4. 注册

```python
# 注册数据集
DatasetRegistry.register(
    name='your_dataset',              # 数据集名称
    loader=load_your_dataset,         # 加载函数
    prompt_template=your_prompt,      # prompt 模板
    description='Dataset description', # 描述
    default_batch_size=128,           # 默认批量大小
    default_max_length=100            # 默认最大生成长度
)

# 注册评估器（可选）
EvaluatorRegistry.register(
    name='your_evaluator',
    evaluator=your_evaluator,
    description='Evaluator description',
    metrics=['accuracy', 'f1', 'etc']
)
```

---

## 📚 更多示例

### 示例 1: GSM8K (数学问题)

```python
def load_gsm8k(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def gsm8k_prompt(item: dict) -> str:
    return f"Question: {item['question']}\n\nAnswer: Let's solve this step by step.\n"

DatasetRegistry.register(
    name='gsm8k',
    loader=load_gsm8k,
    prompt_template=gsm8k_prompt,
    description='GSM8K grade school math',
    default_batch_size=64,
    default_max_length=256
)
```

使用：
```bash
python run_pipeline.py \
    --dataset gsm8k \
    --dataset-path /path/to/gsm8k.jsonl \
    --evaluator generation \
    --batch-size 64 \
    --max-length 256
```

### 示例 2: HumanEval (代码生成)

```python
def load_humaneval(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def humaneval_prompt(item: dict) -> str:
    return f"{item['prompt']}\n"  # HumanEval 已经包含完整 prompt

DatasetRegistry.register(
    name='humaneval',
    loader=load_humaneval,
    prompt_template=humaneval_prompt,
    description='HumanEval code generation',
    default_batch_size=32,
    default_max_length=512
)
```

### 示例 3: 自定义 JSON 数据

假设你的数据格式是：
```json
{
  "examples": [
    {
      "input": "Translate to French: Hello",
      "output": "Bonjour",
      "metadata": {"lang": "fr"}
    }
  ]
}
```

注册：
```python
def load_custom(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
        return data['examples']  # 提取 examples 数组

def custom_prompt(item: dict) -> str:
    return item['input']  # 直接使用 input 字段

DatasetRegistry.register(
    name='my_translation',
    loader=load_custom,
    prompt_template=custom_prompt,
    description='My translation dataset',
    default_batch_size=128,
    default_max_length=50
)
```

---

## ✅ 验证

注册完成后，验证是否成功：

```python
from dataset_registry import DatasetRegistry
from evaluator_registry import EvaluatorRegistry

# 列出所有数据集
print("Datasets:", DatasetRegistry.list_datasets())

# 列出所有评估器
print("Evaluators:", EvaluatorRegistry.list_evaluators())

# 测试加载
data = DatasetRegistry.load_dataset(
    name='mmlu_pro',
    path='/path/to/mmlu_pro.jsonl',
    limit=5  # 只加载5个样本测试
)

print(f"Loaded {len(data)} samples")
print(f"First prompt: {data[0]['prompt'][:100]}...")
```

---

## 🎯 最佳实践

1. **数据加载器**：尽量保持简单，只负责读取和解析
2. **Prompt Template**：根据任务特点设计，可以创建多个版本（标准版、CoT版等）
3. **评估器**：优先使用内置评估器，特殊需求再自定义
4. **命名规范**：使用清晰的名称，如 `dataset_name`, `dataset_name_cot`
5. **测试**：使用 `--limit` 参数先测试小量数据
6. **文档**：在注册时填写清晰的 `description`

---

## 💡 提示

- 一个数据集可以注册多个版本（不同的 prompt）
- 数据集和评估器可以组合使用
- 使用 `limit` 参数快速调试
- 查看 `register_mmlu_pro.py` 获取完整示例

