"""
Datasets Module - 数据集注册模块
自动加载所有数据集注册文件
"""

from .dataset_registry import DatasetRegistry

# 显式导入所有数据集注册脚本以触发注册逻辑
from . import register_mmlu_pro  # noqa: F401
from . import register_aime24  # noqa: F401
from . import register_aime25  # noqa: F401
from . import register_gpqa_main_zeroshot  # noqa: F401
from . import register_mmlu  # noqa: F401
from . import register_mmlu_pro_plus  # noqa: F401
from . import register_mmlu_prox  # noqa: F401
from . import register_asdiv  # noqa: F401
from . import register_gsm_plus  # noqa: F401
from . import register_gsm8k  # noqa: F401
from . import register_ifeval  # noqa: F401
from . import register_ruler  # noqa: F401
from . import register_hendrycks_math  # noqa: F401
# from . import register_minerva_math  # noqa: F401

__all__ = ['DatasetRegistry']
