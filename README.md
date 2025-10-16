# RWKV 评估框架

这是一个完整的 RWKV 模型评估框架，支持本地批量评估和 API 评估两种模式。

## 📁 项目结构

```
rwkveval/
├── local_batch_benchmark/      # 🚀 本地批量评估（推荐用于开发）
│   ├── tasks/                 # 数据集注册模块（原 datasets/）
│   │   ├── dataset_registry.py
│   │   └── register_*.py
│   ├── evaluators/            # 评估器注册模块
│   │   ├── evaluator_registry.py
│   │   └── register_*.py
│   ├── batch_engine.py        # RWKV推理引擎
│   ├── pipeline.py           # 评估管线
│   ├── run_pipeline.py       # 通用命令行入口
│   ├── run_mmlu_pro.py       # MMLU Pro专用脚本
│   ├── example_usage.py      # Python API使用示例
│   └── README.md             # 详细文档
│
├── api_benchmark/             # 🌐 API评估（用于集成lm_eval等工具）
│   ├── api_server.py         # FastAPI服务器
│   ├── api.sh                # 启动脚本
│   ├── batch_engine.py       # RWKV推理引擎
│   ├── run_eval.sh           # lm_eval评估脚本
│   └── test_lambada_api.py   # API测试
│
├── benchmarkdata/             # 评估数据集
│   ├── mmlupro/              # MMLU Pro数据
│   └── ...
│
├── reference/                 # RWKV模型参考实现
│   ├── rwkv7.py              # RWKV-7模型
│   └── utils.py              # Tokenizer等工具
│
├── tools/                     # 工具脚本
│   └── download.sh           # 数据集下载脚本
│
└── README.md                  # 本文件
```

## 🎯 两种评估模式对比

| 特性 | 本地批量评估 | API评估 |
|------|------------|---------|
| **目录** | `local_batch_benchmark/` | `api_benchmark/` |
| **运行方式** | 直接调用推理引擎 | 通过HTTP API |
| **适用场景** | • 快速迭代开发<br>• 添加新数据集<br>• 自定义评估逻辑 | • 与外部工具集成<br>• lm_eval测评<br>• 标准化API接口 |
| **性能** | ⚡ 更快（无网络开销） | 🌐 稍慢（有HTTP开销） |
| **灵活性** | 🔧 高（完全控制） | 📦 中（标准化接口） |
| **扩展性** | ✅ 模块化注册系统 | ✅ OpenAI兼容API |

---

## 🚀 快速开始

### 方式1: 本地批量评估（推荐）

```bash
cd local_batch_benchmark

# 1. 查看可用组件
python3 run_pipeline.py --list

# 2. 运行 MMLU Pro 评估（快速测试）
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64 \
    --limit 100

# 3. 运行完整 MMLU Pro 评估 + 保存推理结果
python3 run_mmlu_pro.py \
    --dataset-path /home/rwkv/Peter/rwkveval/benchmarkdata/mmlupro \
    --batch-size 64 \
    --inferoutput ./results/mmlu_pro_predictions.jsonl

# 4. 运行 LAMBADA 评估
python3 run_pipeline.py \
    --dataset lambada \
    --dataset-path /path/to/lambada_test.jsonl \
    --evaluator perplexity \
    --batch-size 256 \
    --limit 500
```

**重要参数说明：**
- `--limit N`: 只测试前 N 个样本（快速验证）
- `--inferoutput PATH`: 保存每个样本的推理结果到 JSONL 文件
- `--batch-size N`: 调整批处理大小（根据显存）
- `--cot`: 使用 Chain-of-Thought（针对 MMLU-Pro）

### 方式2: API 评估

```bash
cd api_benchmark

# 1. 启动 API 服务器
./api.sh

# 2. 运行 lm_eval 评估（在另一个终端）
./run_eval.sh
```

---

## 📚 本地批量评估 - 详细说明

### 添加新数据集

1. **创建数据集注册文件** `local_batch_benchmark/tasks/register_your_dataset.py`:

```python
from .dataset_registry import DatasetRegistry
import json

def load_your_dataset(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def your_prompt_template(item: dict) -> str:
    return f"Question: {item['question']}\nAnswer:"

DatasetRegistry.register(
    name='your_dataset',
    loader=load_your_dataset,
    prompt_template=your_prompt_template,
    description='Your dataset',
    default_batch_size=128,
    default_max_length=100
)
```

2. **在 `tasks/__init__.py` 中导入**:

```python
from . import register_your_dataset
```

3. **使用**:

```bash
python3 run_pipeline.py \
    --dataset your_dataset \
    --dataset-path /path/to/data.json \
    --evaluator generation
```

### 内置数据集

| 数据集 | 描述 | 评估器 |
|--------|------|--------|
| `lambada` | LAMBADA语言建模 | `perplexity` |
| `mmlu_pro` | MMLU Pro多选题 | `mmlu_pro` |
| `qa` | 通用问答 | `exact_match` |
| `completion` | 文本补全 | `generation` |
| `cot` | Chain-of-Thought | `generation` |
| `math` | 数学问题 | `generation` |

### 内置评估器

| 评估器 | 描述 | 指标 |
|--------|------|------|
| `perplexity` | 困惑度和准确率 | perplexity, accuracy |
| `exact_match` | 精确匹配 | accuracy |
| `generation` | 生成统计 | throughput, tokens |
| `mmlu_pro` | MMLU多选题 | accuracy, subject_acc |

---

## 🌐 API 评估 - 详细说明

### 启动 API 服务器

编辑 `api_benchmark/api.sh` 配置参数：

```bash
python api_server.py \
    --host 192.168.0.82 \
    --port 8000 \
    --llm_path /path/to/model \
    --batch_size 128 \
    --max_tokens 2048
```

然后运行：
```bash
cd api_benchmark
./api.sh
```

### 使用 lm_eval 评估

编辑 `api_benchmark/run_eval.sh`：

```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://192.168.0.82:8000/v1"

lm_eval --model openai-completions \
    --model_args base_url="http://192.168.0.82:8000/v1/completions",model=davinci-002,tokenized_requests=false \
    --tasks lambada_openai \
    --output_path ./eval_results
```

然后运行：
```bash
./run_eval.sh
```

### API 端点

| 端点 | 描述 |
|------|------|
| `POST /v1/completions` | 文本补全（支持logprobs） |
| `POST /v1/chat/completions` | 聊天对话 |
| `GET /v1/models` | 列出模型 |
| `GET /health` | 健康检查 |
| `GET /stats` | 统计信息 |

---

## 💡 使用建议

### 开发阶段
✅ 使用 **本地批量评估** (`local_batch_benchmark/`)
- 快速迭代
- 添加自定义数据集
- 调试评估逻辑

### 生产/集成阶段
✅ 使用 **API 评估** (`api_benchmark/`)
- 与 lm_eval 集成
- 标准化评估流程
- 多客户端访问

---

## 📖 文档

### 本地批量评估
- 📘 `local_batch_benchmark/README.md` - 完整文档
- 🚀 `local_batch_benchmark/QUICK_START.md` - 快速开始（如果存在）
- 📝 `local_batch_benchmark/HOW_TO_ADD_NEW_DATASET.md` - 添加数据集教程（如果存在）

### API 评估
- 📘 `api_benchmark/README.md` - API文档（如果存在）

---

## 🎯 典型工作流程

### 场景1: 添加新的评测任务

1. 在 `local_batch_benchmark/datasets/` 创建数据集注册
2. （可选）在 `local_batch_benchmark/evaluators/` 创建自定义评估器
3. 在 `datasets/__init__.py` 和 `evaluators/__init__.py` 中导入
4. 使用 `run_pipeline.py` 运行评估
5. 验证结果后，可通过 API 模式集成到 lm_eval

### 场景2: 运行标准评测

1. 直接使用 `local_batch_benchmark/run_pipeline.py`
2. 或者使用 `api_benchmark/run_eval.sh` 通过 lm_eval

### 场景3: 大规模评测

1. 启动 API 服务器 (`api_benchmark/api.sh`)
2. 使用 lm_eval 或其他工具连接 API
3. 并发评估多个任务

---

## 🔧 环境配置

```bash
# 设置模型路径（可选）
export MODEL_PATH="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"

# HuggingFace 配置（用于下载数据集）
export HF_TOKEN="your_token"
export HF_HOME="~/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
```

---

## 📊 结果输出

### 本地批量评估
结果保存在 `local_batch_benchmark/eval_results/`:
```
eval_results/
├── lambada_perplexity_20241016_123456.json
├── mmlu_pro_mmlu_pro_20241016_123500.json
└── summary_20241016_123600.json
```

### API 评估
lm_eval 结果保存在 `api_benchmark/eval_results/` 或指定的输出目录

---

## 🤝 贡献

欢迎添加新的数据集和评估器！

1. Fork 项目
2. 在 `local_batch_benchmark/datasets/` 或 `local_batch_benchmark/evaluators/` 添加注册文件
3. 更新 `__init__.py`
4. 提交 Pull Request

---

## 📮 联系

如有问题，请查看各目录下的 README.md 或提issue。

