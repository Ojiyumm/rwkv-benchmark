# RWKV Benchmarking API

高性能RWKV批量推理引擎与OpenAI兼容的API服务器。

## 文件说明

- **batch_engine.py** - RWKV推理引擎（模型加载、batch infer、batch decode）
- **test.py** - 完整测试套件（Token分析、简单生成、随机问题、LAMBADA评估）
- **api_server.py** - FastAPI服务器，提供OpenAI兼容的API接口
- **start_api.sh** - API服务器启动脚本
- **test_api.py** - API服务器测试脚本
- **config.py** - 配置文件

## 快速开始

### 1. 安装依赖

```bash
pip install torch fastapi uvicorn pydantic requests numpy
```

### 2. 运行完整测试套件

运行所有测试（参考batch.py的实现）：

```bash
cd /home/rwkv/Peter/Albatross/benchmarking
python test.py
```

这会运行四个测试：

**Test 1: Next Token Analysis** - 分析下一个token的概率分布
- 参考batch.py第50-61行的实现
- 显示每个prompt的top-k可能token

**Test 2: Simple Generation** - 简单文本生成测试  
- 参考batch.py第65-87行的实现
- 使用正确的token处理方式（squeeze）

**Test 3: Random Questions** - 128个随机问题批量推理
- 从5个问题中随机选择128个
- 批量生成并保存到JSONL
- 显示每个问题的示例输出

**Test 4: LAMBADA Eval** - LAMBADA数据集评估（可选）
- 参考batch.py第90-152行的实现
- 默认注释掉，可以取消注释运行

**预期输出示例：**
```
Test 1: Next Token Analysis
Prompt: The apple can be
  ' a'                 [probability 15.23%]
  ' eaten'             [probability 12.45%]
  ...

Test 2: Simple Generation
也许我们应该考虑一下这个问题。
他们发现了一个新的解决方案...

Test 3: Random Questions
Total samples: 128
Throughput: 5400.12 tokens/s
✓ Saved 128 results to batch_results_20251010_123456.jsonl
```

### 2.1 验证实现一致性

验证batch_engine.py的实现与batch.py完全一致：

```bash
python verify_consistency.py
```

这会用相同的随机种子运行两次，比较结果是否一致。

### 2.2 仅测试引擎（不运行完整测试）

如果只想测试引擎是否正常工作：

```bash
python batch_engine.py
```

## 重要说明

### 关于随机性和确定性

**Noise参数的影响：**
- `noise=0.0`: 确定性采样（greedy），相同输入总是产生相同输出
- `noise>0.0`: 随机采样，每次运行结果会不同

**设置随机种子：**
```python
engine = RWKVInferenceEngine(
    model_path=config.MODEL_PATH,
    seed=42  # 设置随机种子确保可重复性
)
```

即使设置了随机种子，如果`noise>0`，不同运行批次的结果仍会不同。要获得完全确定的结果，需要：
1. 设置相同的随机种子
2. 使用`noise=0`
3. 使用相同的模型实例和状态

### 3. 启动API服务器

```bash
./start_api.sh
```

或者自定义端口：

```bash
PORT=8080 ./start_api.sh
```

服务器会在启动时加载模型，看到以下信息表示成功：

```
Loading model from /home/rwkv/models/rwkv7/...
Model loaded successfully!
Engine initialized successfully!
```

### 4. 测试API服务器

在另一个终端运行API测试脚本：

```bash
python test_api.py
```

或使用curl测试：

```bash
# 健康检查
curl http://localhost:8000/health

# 文本补全
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7-world",
    "prompt": "The quick brown fox",
    "max_tokens": 50
  }'

# Chat补全
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7-world",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'

# 性能统计
curl http://localhost:8000/stats
```

## API端点

### OpenAI兼容端点

- `POST /v1/completions` - 文本补全
- `POST /v1/chat/completions` - 对话补全
- `GET /v1/models` - 列出可用模型

### 额外端点

- `GET /health` - 健康检查
- `GET /stats` - 获取引擎性能统计
- `GET /` - API信息

## 配置

### 修改模型路径

编辑 `api_server.py` 和 `batch_engine.py` 中的模型路径：

```python
model_path="/home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096"
```

### 调整批处理参数

在 `api_server.py` 的 `lifespan` 函数中：

```python
engine = BatchInferenceEngine(
    model_path="...",
    max_batch_size=128,      # 最大批量大小
    max_wait_time=0.01,      # 批次等待时间（秒）
    sampler_noise=3.0        # 采样噪声
)
```

### 环境变量

在 `start_api.sh` 中设置：

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU编号
export HOST=0.0.0.0            # 监听地址
export PORT=8000               # 端口
```

## 与Benchmark工具集成

此API完全兼容OpenAI格式，可直接用于各种benchmark工具：

### 使用vLLM benchmark

```bash
python -m vllm.entrypoints.openai.benchmark \
  --api-url http://localhost:8000/v1 \
  --model rwkv-7-world \
  --num-prompts 100
```

### 使用LLMPerf

```python
from llmperf import LLMPerf

perf = LLMPerf(
    base_url="http://localhost:8000/v1",
    model="rwkv-7-world"
)
results = perf.run_benchmark()
```

### 使用OpenAI Python SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 不需要真实API key
)

response = client.chat.completions.create(
    model="rwkv-7-world",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

## 性能特点

- ✅ **动态Batching**: 自动将并发请求组合成批次
- ✅ **高吞吐量**: 批量推理显著提升吞吐量
- ✅ **低延迟**: 小批次等待时间优化延迟
- ✅ **OpenAI兼容**: 无需修改客户端代码
- ✅ **实时统计**: 通过`/stats`端点监控性能

## 故障排查

### 模型加载失败

检查模型路径是否正确：
```bash
ls -lh /home/rwkv/models/rwkv7/rwkv7-g1-0.4b-20250324-ctx4096.pth
```

### CUDA内存不足

减小批量大小：
```python
max_batch_size=64  # 或更小
```

### 端口已被占用

更改端口：
```bash
PORT=8080 ./start_api.sh
```

## 性能优化建议

1. **调整批量大小**: 根据GPU显存调整`max_batch_size`
2. **调整等待时间**: 低延迟场景减小`max_wait_time`，高吞吐场景增大
3. **使用FP16**: 默认已启用，提供最佳性能
4. **监控统计**: 定期检查`/stats`端点了解系统状态

## 许可证

与RWKV项目相同

