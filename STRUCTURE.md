# 项目结构说明

## 📂 完整目录树

```
rwkveval/
│
├── local_batch_benchmark/          # 本地批量评估模块
│   │
│   ├── datasets/                   # 数据集模块（注册器模式）
│   │   ├── __init__.py            # 自动导入所有数据集
│   │   ├── dataset_registry.py     # 核心注册器
│   │   └── register_mmlu_pro.py   # MMLU Pro 数据集
│   │
│   ├── evaluators/                 # 评估器模块（注册器模式）
│   │   ├── __init__.py            # 自动导入所有评估器
│   │   ├── evaluator_registry.py   # 核心注册器
│   │   └── register_mmlu_pro.py   # MMLU Pro 评估器
│   │
│   ├── batch_engine.py            # RWKV 推理引擎
│   ├── pipeline.py                # 评估管线核心
│   ├── run_pipeline.py            # 命令行入口
│   ├── run_mmlu_pro.py           # MMLU Pro 专用脚本
│   ├── example_usage.py          # 使用示例
│   └── README.md                 # 模块文档
│
├── api_benchmark/                 # API 评估模块
│   ├── api_server.py             # FastAPI 服务器
│   ├── api.sh                    # 启动脚本
│   ├── batch_engine.py           # RWKV 推理引擎
│   ├── run_eval.sh               # lm_eval 评估脚本
│   └── test_lambada_api.py       # API 测试
│
├── reference/                     # RWKV 参考实现
│   ├── rwkv7.py                  # RWKV-7 模型
│   ├── utils.py                  # 工具函数
│   └── rwkv_vocab_v20230424.txt  # 词汇表
│
├── docs/                          # 文档目录
├── tools/                         # 工具脚本
├── README.md                      # 项目主文档
└── STRUCTURE.md                   # 本文件
```

## 🎯 设计理念

### 1. 模块化分离
- **本地批量评估** 和 **API评估** 完全分离
- 互不干扰，各自独立运行

### 2. 注册器模式
- **数据集注册器** (`datasets/`)：管理所有数据集的加载和 prompt 模板
- **评估器注册器** (`evaluators/`)：管理所有评估方法和指标计算

### 3. 自动导入
- 在 `datasets/__init__.py` 和 `evaluators/__init__.py` 中自动导入所有注册文件
- 只需导入模块，所有组件自动可用

## 🔄 工作流程

### 数据集注册流程

```
1. 创建注册文件
   datasets/register_xxx.py
   
2. 定义加载器和模板
   def load_xxx(path) -> List[Dict]
   def xxx_prompt(item) -> str
   
3. 调用注册
   DatasetRegistry.register(name, loader, template, ...)
   
4. 在 __init__.py 导入
   from . import register_xxx
   
5. 自动可用
   DatasetRegistry.list_datasets()  # 包含新数据集
```

### 评估器注册流程

```
1. 创建注册文件
   evaluators/register_xxx.py
   
2. 定义评估函数
   def xxx_evaluator(engine, data, **kwargs) -> Dict
   
3. 调用注册
   EvaluatorRegistry.register(name, evaluator, ...)
   
4. 在 __init__.py 导入
   from . import register_xxx
   
5. 自动可用
   EvaluatorRegistry.list_evaluators()  # 包含新评估器
```

## 📝 文件职责

### local_batch_benchmark/

| 文件 | 职责 |
|------|------|
| `datasets/dataset_registry.py` | 数据集注册器核心，提供注册、获取、加载功能 |
| `datasets/register_*.py` | 具体数据集的注册（加载器 + prompt模板） |
| `datasets/__init__.py` | 自动导入所有数据集注册 |
| `evaluators/evaluator_registry.py` | 评估器注册器核心，提供注册、获取、运行功能 |
| `evaluators/register_*.py` | 具体评估器的注册（评估函数） |
| `evaluators/__init__.py` | 自动导入所有评估器注册 |
| `batch_engine.py` | RWKV推理引擎，提供批量推理能力 |
| `pipeline.py` | 评估管线，串联数据加载、推理、评估 |
| `run_pipeline.py` | 命令行入口，提供友好的CLI |
| `run_*.py` | 特定任务的快捷脚本 |
| `example_usage.py` | Python API 使用示例 |

### api_benchmark/

| 文件 | 职责 |
|------|------|
| `api_server.py` | FastAPI服务器，提供OpenAI兼容的API |
| `api.sh` | API启动脚本 |
| `batch_engine.py` | RWKV推理引擎（与local版本功能相同） |
| `run_eval.sh` | lm_eval评估脚本 |
| `test_*.py` | API测试脚本 |

## 🔗 依赖关系

```
run_pipeline.py
    ├── pipeline.py
    │   ├── batch_engine.py
    │   ├── datasets (module)
    │   │   └── dataset_registry.py
    │   └── evaluators (module)
    │       └── evaluator_registry.py
    └── datasets/__init__.py (自动注册)
    └── evaluators/__init__.py (自动注册)
```

## 🌟 核心类

### DatasetRegistry (datasets/dataset_registry.py)
```python
class DatasetRegistry:
    @classmethod
    def register(cls, name, loader, prompt_template, ...)
    
    @classmethod
    def get(cls, name) -> DatasetConfig
    
    @classmethod
    def load_dataset(cls, name, path) -> List[Dict]
    
    @classmethod
    def list_datasets(cls) -> List[str]
```

### EvaluatorRegistry (evaluators/evaluator_registry.py)
```python
class EvaluatorRegistry:
    @classmethod
    def register(cls, name, evaluator, ...)
    
    @classmethod
    def get(cls, name) -> EvaluatorConfig
    
    @classmethod
    def evaluate(cls, name, engine, data) -> Dict
    
    @classmethod
    def list_evaluators(cls) -> List[str]
```

### EvaluationPipeline (pipeline.py)
```python
class EvaluationPipeline:
    def __init__(self, model_path, ...)
    
    def run(self, dataset_name, dataset_path, evaluator_name, ...) -> Dict
    
    def run_multiple(self, tasks) -> List[Dict]
```

## 💡 添加新组件的步骤

### 添加新数据集

1. 在 `local_batch_benchmark/datasets/` 创建 `register_your_dataset.py`
2. 定义 `load_your_dataset()` 和 `your_prompt_template()`
3. 调用 `DatasetRegistry.register()`
4. 在 `datasets/__init__.py` 添加 `from . import register_your_dataset`
5. 运行 `python run_pipeline.py --list` 验证

### 添加新评估器

1. 在 `local_batch_benchmark/evaluators/` 创建 `register_your_evaluator.py`
2. 定义 `your_evaluator(engine, data, **kwargs)`
3. 调用 `EvaluatorRegistry.register()`
4. 在 `evaluators/__init__.py` 添加 `from . import register_your_evaluator`
5. 运行 `python run_pipeline.py --list` 验证

## 🎨 设计模式

- **注册器模式**: 集中管理数据集和评估器
- **模板方法模式**: 定义评估流程的框架
- **策略模式**: 可插拔的数据加载器和评估器
- **工厂模式**: 通过名称创建数据集和评估器实例

## 📦 包组织

```python
# 用户代码
from datasets import DatasetRegistry      # 自动注册所有数据集
from evaluators import EvaluatorRegistry  # 自动注册所有评估器
from pipeline import quick_eval           # 快速评估接口

# 运行评估
results = quick_eval(
    model_path="/path/to/model",
    dataset_name="lambada",        # 已注册的数据集名称
    dataset_path="/path/to/data",
    evaluator_name="perplexity"    # 已注册的评估器名称
)
```

## 🔍 查找指南

需要... | 查找位置
--------|----------
添加新数据集 | `local_batch_benchmark/datasets/register_*.py`
添加新评估器 | `local_batch_benchmark/evaluators/register_*.py`
修改推理逻辑 | `batch_engine.py`
修改评估流程 | `pipeline.py`
修改CLI | `run_pipeline.py`
API服务器 | `api_benchmark/api_server.py`
lm_eval集成 | `api_benchmark/run_eval.sh`

## 🎯 最佳实践

1. **数据集命名**: 使用小写下划线，如 `lambada`, `mmlu_pro`
2. **评估器命名**: 描述性名称，如 `perplexity`, `exact_match`
3. **文件命名**: `register_<name>.py` 表示注册文件
4. **导入顺序**: 先导入基类，再导入注册
5. **错误处理**: 在 `__init__.py` 中使用 try-except 捕获导入错误

## 📚 相关文档

- `README.md` - 项目总览
- `local_batch_benchmark/README.md` - 本地评估详细文档
- `example_usage.py` - 代码示例

