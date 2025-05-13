"""
训练模块包含所有模型训练相关脚本
"""

# 导入train_best_graphsage.py中的main函数
from .train_best_graphsage import main as train_best_graphsage_main

# 如果存在train_light_graphsage.py，也导入它的main函数
try:
    from .train_light_graphsage import main as train_light_graphsage_main
except ImportError:
    train_light_graphsage_main = None

# 为向后兼容性，保留原始别名
train_main = train_best_graphsage_main

# 导出的符号
__all__ = [
    'train_main',  
    'train_best_graphsage_main',
]

# 如果存在light版本，也导出它
if train_light_graphsage_main is not None:
    __all__.append('train_light_graphsage_main') 