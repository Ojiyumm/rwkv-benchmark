"""
Evaluators Module - 评估器注册模块
自动加载所有评估器注册文件
"""

from .evaluator_registry import EvaluatorRegistry

# 导入所有评估器注册
try:
    from . import register_mmlu_pro
except ImportError as e:
    print(f"Warning: Could not import register_mmlu_pro: {e}")

# 可以继续添加更多评估器
# from . import register_custom_evaluator

__all__ = ['EvaluatorRegistry']

