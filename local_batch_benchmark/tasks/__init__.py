"""
Datasets Module - 数据集注册模块
自动加载所有数据集注册文件
"""

from .dataset_registry import DatasetRegistry

# 导入所有数据集注册
try:
    from . import register_mmlu_pro
except ImportError as e:
    print(f"Warning: Could not import register_mmlu_pro: {e}")

# 可以继续添加更多数据集
# from . import register_gsm8k
# from . import register_humaneval

__all__ = ['DatasetRegistry']
