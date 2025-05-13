"""
配置文件，包含所有必要的配置参数
"""

import os
from typing import Dict
import torch

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据预处理配置
DATA_PREPROCESSING: Dict = {
    'missing_values_strategy': 'mean',  # 处理缺失值的策略：mean, median, mode, constant
    'outliers_strategy': 'clip',  # 处理异常值的策略：clip
    'encoding_method': 'label',  # 编码方法：one-hot, label
    'normalization_method': 'standard',  # 标准化方法：standard
    'test_size': 0.15,  # 测试集比例
    'val_size': 0.15,  # 验证集比例
    'random_state': 42,  # 随机种子
    'pca_components': 100,  # PCA降维后的特征数
    'feature_enhancement': True  # 是否增强特征
}

# 网络构建配置
NETWORK_CONSTRUCTION: Dict = {
    'augmentation': True,  # 是否进行数据增强
    'noise_std': 0.1,  # 高斯噪声标准差
    'edge_perturb_prob': 0.1,  # 边扰动概率
    'num_negative_samples': 2,  # 负样本数量倍数
    'feature_enhancement': True,  # 是否进行特征增强
    'edge_perturbation_ratio': 0.2  # 边扰动比例
}

# 训练配置
TRAINING_CONFIG: Dict = {
    'num_epochs': 100,  # 训练轮数
    'learning_rate': 0.01,  # 学习率
    'weight_decay': 0.0001,  # 权重衰减
    'batch_size': 32,  # 批次大小
    'early_stopping_patience': 10,  # 早停耐心值
    'pos_weight': 1.0,  # 正样本权重
    'optimizer': 'adam',  # 优化器类型
    'scheduler': 'reduce_lr_on_plateau',  # 学习率调度器
    'scheduler_patience': 5,  # 调度器耐心值
    'scheduler_factor': 0.5,  # 学习率衰减因子
    'grad_clip': 1.0,  # 梯度裁剪阈值
    'save_best_only': True,  # 是否只保存最佳模型
    'save_freq': 10  # 模型保存频率（轮数）
}

# 模型配置
MODEL_CONFIG: Dict = {
    'input_dim': 3,  # 输入特征维度
    'hidden_dim': 64,  # 隐藏层维度
    'output_dim': 32,  # 输出层维度
    'num_layers': 3,  # 图卷积层数
    'dropout': 0.5,  # Dropout率
    'num_epochs': 100,  # 训练轮数
    'learning_rate': 0.01,  # 学习率
    'early_stopping_patience': 10  # 早停耐心值
}

# 评估配置
EVALUATION_CONFIG: Dict = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],  # 评估指标
    'threshold': 0.5,  # 分类阈值
    'save_predictions': True,  # 是否保存预测结果
    'save_metrics': True,  # 是否保存评估指标
    'plot_curves': True  # 是否绘制曲线
}

# 可视化配置
VISUALIZATION_CONFIG: Dict = {
    'network': {
        'node_size': 100,  # 节点大小
        'edge_width': 1.0,  # 边宽度
        'node_color': 'lightblue',  # 节点颜色
        'edge_color': 'gray'  # 边颜色
    },
    'training': {
        'plot_metrics': True,  # 是否绘制指标
        'save_plots': True  # 是否保存图表
    },
    'plot_training_curves': True,
    'plot_confusion_matrix': True,
    'plot_roc_curve': True,
    'plot_network': True,
    'plot_embeddings': {
        'enabled': True,
        'method': 'tsne',  # 可选: "tsne", "umap", "pca"
        'perplexity': 30,  # 用于t-SNE
        'n_neighbors': 15  # 用于UMAP
    },
    'dpi': 300,
    'figure_size': (10, 6),
    'style': 'seaborn'  # 可选: "seaborn", "ggplot", "classic"
}

# 日志配置
LOGGING_CONFIG: Dict = {
    'level': 'INFO',  # 日志级别
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(BASE_DIR, 'app.log')
} 