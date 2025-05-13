#!/usr/bin/env python3
"""
改进版分类学图GraphSAGE模型训练脚本
重点解决过拟合问题，增加正则化，并添加特征选择和数据增强
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from torch_geometric.transforms import RandomNodeSplit

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def feature_selection(data, threshold=0.01):
    """
    使用方差阈值进行特征选择，过滤掉方差较小的特征
    """
    logger.info(f"原始特征维度: {data.x.size(1)}")
    
    # 转换为numpy以便使用sklearn
    X = data.x.cpu().numpy()
    
    # 应用方差阈值选择
    selector = VarianceThreshold(threshold)
    try:
        X_new = selector.fit_transform(X)
        indices = selector.get_support(indices=True)
        
        # 更新特征
        data.x = torch.tensor(X_new, dtype=data.x.dtype, device=data.x.device)
        logger.info(f"特征选择后维度: {data.x.size(1)}")
        logger.info(f"保留了 {len(indices)}/{X.shape[1]} 特征 ({len(indices)/X.shape[1]*100:.1f}%)")
        
        return data
    except Exception as e:
        logger.error(f"特征选择失败: {str(e)}")
        return data

def edge_dropout(data, dropout_rate=0.1):
    """
    随机丢弃一部分边来增强数据，减少过拟合
    """
    num_edges = data.edge_index.size(1)
    mask_size = int((1 - dropout_rate) * num_edges)
    
    # 随机选择要保留的边
    perm = torch.randperm(num_edges)
    mask = perm[:mask_size]
    
    # 应用掩码
    data.edge_index = data.edge_index[:, mask]
    logger.info(f"边数量: {num_edges} -> {data.edge_index.size(1)} (丢弃率={dropout_rate:.2f})")
    
    return data

def create_balanced_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    创建分层的数据划分，保持每个类别在训练/验证/测试集中的比例一致
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"
    
    # 获取类别数量
    num_classes = int(data.y.max().item()) + 1
    num_nodes = data.x.size(0)
    
    # 初始化掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 按类别分割
    for c in range(num_classes):
        # 获取该类别的所有节点
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        num_class_nodes = idx.size(0)
        
        # 打乱顺序
        perm = torch.randperm(num_class_nodes)
        idx = idx[perm]
        
        # 划分
        train_size = int(num_class_nodes * train_ratio)
        val_size = int(num_class_nodes * val_ratio)
        
        # 设置掩码
        train_mask[idx[:train_size]] = True
        val_mask[idx[train_size:train_size+val_size]] = True
        test_mask[idx[train_size+val_size:]] = True
    
    # 更新数据对象
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    logger.info(f"训练集: {train_mask.sum().item()}个节点 ({train_mask.sum().item()/num_nodes*100:.1f}%)")
    logger.info(f"验证集: {val_mask.sum().item()}个节点 ({val_mask.sum().item()/num_nodes*100:.1f}%)")
    logger.info(f"测试集: {test_mask.sum().item()}个节点 ({test_mask.sum().item()/num_nodes*100:.1f}%)")
    
    # 验证每个类别的分布
    for c in range(num_classes):
        train_count = (data.y[train_mask] == c).sum().item()
        val_count = (data.y[val_mask] == c).sum().item()
        test_count = (data.y[test_mask] == c).sum().item()
        total = train_count + val_count + test_count
        
        logger.info(f"类别 {c}: 训练={train_count}/{total} ({train_count/total*100:.1f}%), "
                  f"验证={val_count}/{total} ({val_count/total*100:.1f}%), "
                  f"测试={test_count}/{total} ({test_count/total*100:.1f}%)")
    
    return data

def plot_training_curve(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"训练曲线已保存到: {save_path}")

def main():
    """主函数"""
    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 尝试导入模型
    try:
        from models.best_graphsage_model import GraphSAGE, train_model, evaluate_model
        print("成功导入 best_graphsage_model")
    except ImportError:
        try:
            from models.light_graphsage_model import GraphSAGE, train_model, evaluate_model
            print("成功导入 light_graphsage_model")
        except ImportError:
            try:
                # 尝试添加父目录
                parent_dir = os.path.dirname(current_dir)
                sys.path.append(parent_dir)
                sys.path.append(os.path.join(parent_dir, 'models'))
                from models.best_graphsage_model import GraphSAGE, train_model, evaluate_model
                print("从父目录导入 best_graphsage_model")
            except ImportError:
                from best_graphsage_model import GraphSAGE, train_model, evaluate_model
                print("从当前目录导入 best_graphsage_model")
    
    # 尝试加载数据
    data_paths = [
        os.path.join('data', 'taxonomy_graph_data.pt'),  # 相对路径
        os.path.join(current_dir, 'data', 'taxonomy_graph_data.pt'),  # 从当前目录
        os.path.join(os.path.dirname(current_dir), 'data', 'taxonomy_graph_data.pt'),  # 从父目录
        "M:\\4.9\\data\\taxonomy_graph_data.pt"  # 绝对路径
    ]
    
    data = None
    for path in data_paths:
        logger.info(f"尝试从路径加载: {path}")
        if os.path.exists(path):
            try:
                data = torch.load(path)
                logger.info(f"成功从 {path} 加载数据")
                break
            except Exception as e:
                logger.error(f"从 {path} 加载失败: {str(e)}")
    
    if data is None:
        raise FileNotFoundError("找不到分类学图数据文件")
    
    # 打印原始数据信息
    logger.info(f"原始节点特征维度: {data.x.shape}")
    logger.info(f"原始边索引维度: {data.edge_index.shape}")
    
    if hasattr(data, 'y'):
        logger.info(f"标签维度: {data.y.shape}")
        
        # 分析标签
        if data.y.dim() == 1:
            num_classes = int(data.y.max().item() + 1)
            class_counts = [(data.y == i).sum().item() for i in range(num_classes)]
            logger.info(f"类别数量: {num_classes}")
            logger.info(f"类别分布: {class_counts}")
    else:
        logger.info("数据中没有标签")
    
    # ========== 数据预处理 ==========
    # 1. 特征选择，减少维度
    data = feature_selection(data, threshold=0.005)
    
    # 2. 创建分层的训练/验证/测试划分
    data = create_balanced_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # 3. 对训练集应用边丢弃增强
    if torch.rand(1).item() < 0.5:  # 50%概率应用边丢弃
        data = edge_dropout(data, dropout_rate=0.05)
    
    # ========== 模型配置 ==========
    # 设置更适合的模型参数
    hidden_channels = 64   # 减小隐藏层，降低过拟合风险
    num_layers = 2         # 减少层数，防止过度学习
    dropout = 0.3          # 增加Dropout，加强正则化
    learning_rate = 0.001  # 降低学习率
    weight_decay = 1e-3    # 增加权重衰减，加强L2正则化
    epochs = 300           # 增加轮数，但依赖早停
    patience = 30          # 更长的早停耐心值
    
    # 初始化模型
    out_channels = 1 if data.y.dim() == 1 or data.y.size(1) == 1 else data.y.size(1)
    if data.y.dim() == 1:
        out_channels = int(data.y.max().item() + 1)
    
    logger.info(f"初始化模型: 输入={data.x.size(1)}, 隐藏={hidden_channels}, 输出={out_channels}, 层数={num_layers}")
    
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # 移动数据到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    model = model.to(device)
    data = data.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 创建学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate/10
    )
    
    # 损失函数 - 使用带权重的交叉熵
    if hasattr(data, 'y') and data.y is not None:
        num_classes = int(data.y.max().item() + 1)
        # 计算类别权重
        class_counts = torch.zeros(num_classes, device=device)
        for i in range(num_classes):
            class_counts[i] = (data.y == i).sum().item()
        
        # 反比权重
        class_weights = class_counts.sum() / (class_counts * num_classes)
        logger.info(f"类别权重: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ========== 训练过程 ==========
    logger.info("开始训练...")
    model.train()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # L1正则化
        l1_lambda = 1e-5
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算训练准确率
        _, pred = out.max(dim=1)
        correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = float(correct.sum()) / int(data.train_mask.sum())
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            
            # 计算验证准确率
            _, pred = out.max(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = float(correct.sum()) / int(data.val_mask.sum())
        
        # 记录指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 更新学习率调度器
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch: {epoch+1:03d}, LR: {current_lr:.6f}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停检查 - 使用验证准确率作为指标
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            logger.info(f"发现新的最佳模型! 验证准确率: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"早停! {patience}轮没有改进")
            break
    
    # 绘制训练曲线
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'training_curves.png')
    plot_training_curve(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试阶段
    logger.info("测试模型...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        
        # 记录每个类别的准确率
        correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = float(correct.sum()) / int(data.test_mask.sum())
        logger.info(f"测试准确率: {test_acc:.4f}")
        
        # 每个类别的准确率
        for c in range(out_channels):
            mask = (data.y == c) & data.test_mask
            if mask.sum() > 0:
                class_correct = float((pred[mask] == c).sum()) / int(mask.sum())
                logger.info(f"类别 {c} 准确率: {class_correct:.4f} ({int((pred[mask] == c).sum())}/{int(mask.sum())})")
    
    # 保存模型
    weights_dir = os.path.join(os.path.dirname(current_dir), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, 'taxonomy_graphsage_improved.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': data.x.size(1),
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'train_metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }
    }, model_path)
    
    logger.info(f"模型已保存到 {model_path}")
    
    return model, test_acc

if __name__ == "__main__":
    try:
        logger.info("开始改进版分类学图GraphSAGE训练...")
        model, test_acc = main()
        logger.info(f"训练完成! 最终测试准确率: {test_acc:.4f}")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc() 