import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention机制"""
    def __init__(self, in_channels, reduction=4):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取输入维度
        batch_size, channels = x.size()
        
        # 调整为1D平均池化所需的形状 (batch_size, channels, 1)
        x_reshaped = x.view(batch_size, channels)
        
        # Squeeze - 全局平均池化
        y = torch.mean(x_reshaped, dim=0).view(1, -1)  # 按特征维度平均
        
        # Excitation - 通过FC层生成权重
        y = self.fc(y)
        
        # 确保y的维度与x匹配
        if batch_size > 1:  # 处理批次大小大于1的情况
            y = y.repeat(batch_size, 1)
        
        # Scale - 返回加权的特征
        return x_reshaped * y

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 增强特征提取和表示能力
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        # 输入投影层 - 增加初始特征变换
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 第一层
        self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels*2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels*2))
        self.attention_layers.append(SEAttention(hidden_channels*2))
        
        # 隐藏层 - 使用更大的隐藏维度
        hidden_dims = [hidden_channels*2] * (num_layers-2)
        for i in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_dims[i], hidden_dims[i]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i]))
            self.attention_layers.append(SEAttention(hidden_dims[i]))
        
        # 输出层 - 添加额外的非线性变换
        self.conv_layers.append(SAGEConv(hidden_dims[-1] if num_layers > 2 else hidden_channels*2, hidden_channels))
        self.final_proj = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self._init_weights()
        logging.info(f"初始化高性能GraphSAGE: {num_layers}层, 隐藏维度={hidden_channels*2}")
    
    def _init_weights(self):
        """使用更好的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index):
        # 应用输入投影
        h = self.input_proj(x)
        
        # 残差连接列表
        residuals = [h]
        
        # 应用图卷积层
        for i in range(self.num_layers - 1):
            # 图卷积
            h = self.conv_layers[i](h, edge_index)
            
            # 批量归一化
            h = self.batch_norms[i](h)
            
            # 注意力机制
            h = self.attention_layers[i](h)
            
            # 激活和dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # 收集残差
            residuals.append(h)
            
            # 多尺度残差连接 (从之前所有层收集特征)
            if i > 0 and residuals[i].size(1) == h.size(1):
                h = h + residuals[i]
        
        # 最后一层图卷积
        h = self.conv_layers[-1](h, edge_index)
        
        # 应用最终投影
        h = self.final_proj(h)
        
        return h

def train_model(model, data, optimizer, criterion, scheduler=None, num_epochs=200, patience=20, lr_scheduler_type='plateau'):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 存储训练指标
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    
    # 检查数据是否平衡
    if hasattr(data, 'y') and data.y is not None:
        num_classes = data.y.max().item() + 1
        class_counts = [(data.y == i).sum().item() for i in range(num_classes)]
        device = data.x.device  # 获取数据当前的设备
        class_weights = torch.tensor([1.0/c for c in class_counts], device=device)
        weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"类别分布: {class_counts}, 使用加权损失")
    else:
        weighted_criterion = criterion
    
    logging.info(f"{'启用' if use_amp else '禁用'}混合精度训练")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播 - 使用混合精度计算
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(data.x, data.edge_index)
                
                # 确保训练掩码中有节点
                if data.train_mask.sum() == 0:
                    logging.warning("训练掩码中没有节点！创建默认掩码...")
                    data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                
                loss = weighted_criterion(out[data.train_mask], data.y[data.train_mask])
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(data.x, data.edge_index)
            
            # 确保训练掩码中有节点
            if data.train_mask.sum() == 0:
                logging.warning("训练掩码中没有节点！创建默认掩码...")
                data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            
            loss = weighted_criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        
        # 计算训练准确率
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        metrics['lr'].append(current_lr)
        
        # 更新学习率调度器
        if scheduler is not None:
            if lr_scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif lr_scheduler_type == 'cosine':
                scheduler.step()
        
        # 保存指标
        metrics['train_loss'].append(loss.item())
        metrics['val_loss'].append(val_loss.item())
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}, best val_acc: {max(metrics['val_acc']):.4f}")
                break
        
        model.train()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, metrics

def evaluate_model(model, data):
    model.eval()
    
    # 检查必要的属性是否存在
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask') or not hasattr(data, 'y'):
        logging.warning("Missing necessary attributes for evaluation")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # 获取设备
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 使用半精度计算如果在GPU上
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index)
            
        pred = out.argmax(dim=1)
        
        # 计算各个集合的评估指标
        results = {}
        
        # 训练集评估
        train_pred = pred[data.train_mask].cpu().numpy()
        train_true = data.y[data.train_mask].cpu().numpy()
        train_metrics = {
            'accuracy': accuracy_score(train_true, train_pred),
            'precision': precision_score(train_true, train_pred, average='weighted', zero_division=0),
            'recall': recall_score(train_true, train_pred, average='weighted', zero_division=0),
            'f1': f1_score(train_true, train_pred, average='weighted', zero_division=0)
        }
        
        # 验证集评估
        val_pred = pred[data.val_mask].cpu().numpy()
        val_true = data.y[data.val_mask].cpu().numpy()
        val_metrics = {
            'accuracy': accuracy_score(val_true, val_pred),
            'precision': precision_score(val_true, val_pred, average='weighted', zero_division=0),
            'recall': recall_score(val_true, val_pred, average='weighted', zero_division=0),
            'f1': f1_score(val_true, val_pred, average='weighted', zero_division=0)
        }
        
        # 测试集评估
        test_pred = pred[data.test_mask].cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        test_metrics = {
            'accuracy': accuracy_score(test_true, test_pred),
            'precision': precision_score(test_true, test_pred, average='weighted', zero_division=0),
            'recall': recall_score(test_true, test_pred, average='weighted', zero_division=0),
            'f1': f1_score(test_true, test_pred, average='weighted', zero_division=0)
        }
        
        # 尝试计算AUC，如果是二分类问题
        if len(np.unique(train_true)) == 2:
            try:
                train_prob = F.softmax(out[data.train_mask], dim=1)[:, 1].cpu().numpy()
                val_prob = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
                test_prob = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
                
                train_metrics['auc'] = roc_auc_score(train_true, train_prob)
                val_metrics['auc'] = roc_auc_score(val_true, val_prob)
                test_metrics['auc'] = roc_auc_score(test_true, test_prob)
            except:
                logging.warning("无法计算AUC分数，可能是标签分布问题")
        
        # 输出详细评估信息
        logging.info(f"训练集准确率: {train_metrics['accuracy']:.4f}")
        logging.info(f"验证集准确率: {val_metrics['accuracy']:.4f}")
        logging.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        
    return train_metrics, val_metrics, test_metrics 