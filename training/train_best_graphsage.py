import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import logging
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import add_self_loops, remove_self_loops

# 添加父目录到路径，以便导入models中的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.best_graphsage_model import GraphSAGE, train_model, evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enhance_graph_data(data, augment=True):
    """增强图数据，添加自环和更多特征"""
    logging.info(f"原始数据: {data.num_nodes}节点, {data.edge_index.size(1)}边, {data.x.size(1)}特征维度")
    
    # 添加自环连接 - 促进信息传递
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index
    
    if augment:
        # 计算节点度作为额外特征
        node_degrees = torch.zeros(data.num_nodes, dtype=torch.float)
        for i in range(data.edge_index.size(1)):
            node_degrees[data.edge_index[0, i]] += 1
        
        # 节点度归一化
        node_degrees = node_degrees / node_degrees.max()
        
        # 子图影响力 - 简单估计为归一化的度
        node_influence = (node_degrees - node_degrees.mean()) / node_degrees.std()
        node_influence = torch.sigmoid(node_influence).unsqueeze(1)
        
        # 将额外特征与现有特征连接
        augmented_features = torch.cat([
            data.x,
            node_degrees.unsqueeze(1),
            node_influence
        ], dim=1)
        
        data.x = augmented_features
    
    logging.info(f"增强后数据: {data.num_nodes}节点, {data.edge_index.size(1)}边, {data.x.size(1)}特征维度")
    return data

def generate_meaningful_labels(data, strategy='complex'):
    """基于节点特征和图结构生成有意义的标签"""
    if strategy == 'simple':
        # 基本标签生成 - 只基于度和特征均值
        degrees = torch.zeros(data.num_nodes, dtype=torch.float)
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i]
            dst = data.edge_index[1, i]
            degrees[src] += 1
            degrees[dst] += 1
        
        feature_means = data.x.mean(dim=1)
        
        labels = torch.zeros(data.num_nodes, dtype=torch.long)
        threshold_degree = degrees.mean()
        threshold_feature = feature_means.mean()
        
        labels[(degrees > threshold_degree) & (feature_means > threshold_feature)] = 1
    
    else:
        # 复杂标签生成 - 多个因素综合考虑
        # 1. 计算节点度
        degrees = torch.zeros(data.num_nodes, dtype=torch.float)
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i]
            dst = data.edge_index[1, i]
            degrees[src] += 1
            degrees[dst] += 1
        
        # 2. 计算特征统计
        feature_mean = data.x.mean(dim=1)
        feature_std = data.x.std(dim=1)
        feature_max = data.x.max(dim=1)[0]
        
        # 3. 计算简单聚类特征
        from sklearn.cluster import KMeans
        n_clusters = 2  # 二分类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # 确保数据转换为numpy数组并且是浮点型
        cluster_features = data.x.cpu().numpy().astype(np.float64)
        cluster_labels = kmeans.fit_predict(cluster_features)
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
        
        # 4. 基于多个特征生成标签
        normalized_degrees = (degrees - degrees.mean()) / degrees.std()
        normalized_means = (feature_mean - feature_mean.mean()) / feature_mean.std()
        combined_score = normalized_degrees + normalized_means
        
        # 根据分数分配标签
        median_score = torch.median(combined_score)
        labels = (combined_score > median_score).long()
        
        # 使用KMeans标签进行校正 - 增加多样性
        labels = (labels + cluster_labels) % 2
    
    class_counts = [(labels == i).sum().item() for i in range(2)]
    logging.info(f"生成标签分布: 类别0: {class_counts[0]}, 类别1: {class_counts[1]}")
    return labels

def plot_training_metrics(metrics, save_path=None):
    """绘制训练指标"""
    plt.figure(figsize=(15, 10))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 学习率
    if 'lr' in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Metrics plot saved to {save_path}")
    plt.show()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"CUDA可用设备: {torch.cuda.get_device_name(0)}")
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # 获取根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载数据
    data_path = os.path.join(root_dir, 'data', 'graph_data.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = torch.load(data_path)
    logging.info(f"Loaded data with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
    
    # 增强图数据
    data = enhance_graph_data(data, augment=True)
    
    # 打印关键维度以便调试
    logging.info(f"数据特征维度: {data.x.size(1)}")
    
    # 生成复杂有意义的标签
    data.y = generate_meaningful_labels(data, strategy='complex')
    
    # 创建更好的训练、验证和测试掩码
    # 手动创建掩码，避免使用RandomNodeSplit类的版本兼容性问题
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    # 按照7:1.5:1.5的比例划分数据
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    
    # 创建掩码
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 填充掩码
    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True
    
    # 打印掩码信息
    logging.info(f"训练集: {data.train_mask.sum().item()}节点")
    logging.info(f"验证集: {data.val_mask.sum().item()}节点")
    logging.info(f"测试集: {data.test_mask.sum().item()}节点")
    
    # 配置训练参数 - 更高性能设置
    hidden_channels = 128   # 减小隐藏层大小，避免过拟合
    num_layers = 3          # 减少层数，避免过深网络训练不稳定
    dropout = 0.3           # 保持较高的dropout
    learning_rate = 0.005   # 增大学习率，加速收敛
    weight_decay = 5e-4     # 增大权重衰减，增强正则化
    num_epochs = 150        # 减少训练轮次，加快训练
    patience = 20           # 减少早停耐心值
    
    # 初始化高性能模型
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=2,  # 二分类问题
        num_layers=num_layers,
        dropout=dropout
    )
    
    # 检测并使用可用的GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 将模型和数据转移到GPU
    model = model.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    
    # 打印模型架构
    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 使用AdamW优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 使用余弦退火学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练模型
    criterion = nn.CrossEntropyLoss()
    
    # 尝试分批处理大图，避免内存问题
    batch_size = 2048  # 每批处理的节点数量
    if data.num_nodes > batch_size and torch.cuda.is_available():
        logging.info(f"节点数量较大({data.num_nodes})，使用分批处理")
        # 在这里可以添加分批处理逻辑，但GraphSAGE模型通常可以处理整图

    model, metrics = train_model(model, data, optimizer, criterion, scheduler=scheduler, 
                                num_epochs=num_epochs, patience=patience, lr_scheduler_type='cosine')
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"训练后GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # 绘制训练指标
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_training_metrics(metrics, save_path=os.path.join(results_dir, 'training_metrics.png'))
    
    # 最终模型评估
    train_metrics, val_metrics, test_metrics = evaluate_model(model, data)
    
    logging.info("======== 最终模型性能 ========")
    logging.info(f"训练集准确率: {train_metrics['accuracy']:.4f}")
    logging.info(f"验证集准确率: {val_metrics['accuracy']:.4f}")
    logging.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    
    logging.info(f"训练集F1分数: {train_metrics['f1']:.4f}")
    logging.info(f"验证集F1分数: {val_metrics['f1']:.4f}")  
    logging.info(f"测试集F1分数: {test_metrics['f1']:.4f}")
    
    # 保存模型
    weights_dir = os.path.join(root_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, 'best_graphsage_model.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # 保存评估指标
    metrics_path = os.path.join(results_dir, 'metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write(f"metric,value\n")
        f.write(f"train_accuracy,{train_metrics['accuracy']:.4f}\n")
        f.write(f"validation_accuracy,{val_metrics['accuracy']:.4f}\n")
        f.write(f"test_accuracy,{test_metrics['accuracy']:.4f}\n")
        f.write(f"train_f1,{train_metrics['f1']:.4f}\n")
        f.write(f"validation_f1,{val_metrics['f1']:.4f}\n")
        f.write(f"test_f1,{test_metrics['f1']:.4f}\n")
    logging.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 