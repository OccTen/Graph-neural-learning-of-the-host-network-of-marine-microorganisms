#!/usr/bin/env python3
"""
简化版分类学图GraphSAGE模型训练脚本
这个脚本直接使用标准GraphSAGE参数，避免使用不支持的参数
优化了类别平衡和训练过程
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

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
    
    # 使用英文避免中文字体问题
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training curves saved to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")

def main():
    """主函数"""
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
    
    # 打印数据信息
    logger.info(f"节点特征维度: {data.x.shape}")
    logger.info(f"边索引维度: {data.edge_index.shape}")
    
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
    
    # 创建分层的训练/验证/测试划分
    data = create_balanced_split(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    
    # 设置模型参数
    hidden_channels = 64   # 减小隐藏层
    num_layers = 2         # 减少层数
    dropout = 0.2          # 保持适中的Dropout
    learning_rate = 0.001  # 降低学习率
    weight_decay = 5e-4    # 适中的权重衰减
    epochs = 400           # 增加训练轮数
    patience = 50          # 增加早停耐心值
    batch_size = None      # 全批量训练
    
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
    
    # 计算类别权重
    if hasattr(data, 'y') and data.y is not None:
        # 计算训练集中的类别权重（更准确）
        train_y = data.y[data.train_mask]
        class_samples = [(train_y == i).sum().item() for i in range(out_channels)]
        # 使用反比例权重，但限制最大权重
        max_weight = 3.0  # 限制最大权重防止过度关注稀有类
        weights = [min(max(1.0, len(train_y) / (c * out_channels)), max_weight) for c in class_samples]
        class_weights = torch.tensor(weights, device=device)
        logger.info(f"类别权重: {weights}")
    else:
        class_weights = None
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 训练模型
    logger.info("开始训练...")
    model.train()
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
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
            
            # 计算每个类别的验证准确率
            if epoch % 20 == 0:
                for c in range(out_channels):
                    mask = (data.y == c) & data.val_mask
                    if mask.sum() > 0:
                        class_correct = (pred[mask] == c).sum().item()
                        class_total = mask.sum().item()
                        logger.info(f"验证集类别 {c} 准确率: {class_correct / class_total:.4f} ({class_correct}/{class_total})")
        
        # 记录指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 更新学习率调度器 - 使用验证准确率
        scheduler.step(val_acc)
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            logger.info(f'Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
        
        # 计算整体测试准确率
        correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = float(correct.sum()) / int(data.test_mask.sum())
        logger.info(f"测试准确率: {test_acc:.4f}")
        
        # 计算每个类别的测试准确率
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        # 绘制混淆矩阵
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, [f"类别{i}" for i in range(out_channels)], cm_path)
        
        # 打印详细分类报告
        logger.info("\n" + classification_report(y_true, y_pred, digits=4))
        
        # 每个类别的测试准确率
        for c in range(out_channels):
            mask = (data.y == c) & data.test_mask
            if mask.sum() > 0:
                class_correct = (pred[mask] == c).sum().item()
                class_total = mask.sum().item()
                logger.info(f"测试集类别 {c} 准确率: {class_correct / class_total:.4f} ({class_correct}/{class_total})")
    
    # 创建保存目录
    base_dir = "M:\\4.9"
    weights_dir = os.path.join(base_dir, 'weights')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"训练曲线已保存到: {save_path}")
    
    # 保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"混淆矩阵已保存到: {save_path}")
    
    # 保存模型
    model_path = os.path.join(weights_dir, 'taxonomy_graphsage_optimized.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channels': data.x.size(1),
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'num_classes': out_channels
        },
        'train_metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        },
        'class_weights': class_weights.cpu().tolist() if class_weights is not None else None
    }, model_path)
    
    logger.info(f"模型已保存到 {model_path}")
    
    return model, test_acc

if __name__ == "__main__":
    try:
        logger.info("开始优化版分类学图GraphSAGE训练...")
        model, test_acc = main()
        logger.info(f"训练完成! 测试准确率: {test_acc:.4f}")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc() 