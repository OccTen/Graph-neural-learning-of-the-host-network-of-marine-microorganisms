"""
GraphSAGE模型实现 - 高级性能版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv as PYGSAGEConv
from typing import List, Optional, Dict, Any, Union
import logging
import math
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力层"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y.expand_as(x)

class AttentionLayer(nn.Module):
    """多头注意力层，用于聚合邻居特征"""
    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 1):
        super(AttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        
        # 多头注意力
        self.query = nn.Linear(in_channels, self.head_dim * num_heads)
        self.key = nn.Linear(in_channels, self.head_dim * num_heads)
        self.value = nn.Linear(in_channels, self.head_dim * num_heads)
        
        # 输出投影 - 修正为投影到out_channels，确保与卷积层输出维度匹配
        self.output_proj = nn.Linear(self.head_dim * num_heads, out_channels)
        
        # 添加SE注意力
        self.se = SELayer(out_channels, reduction=4)
        
        # 添加门控机制
        self.gate = nn.Linear(in_channels + out_channels, 1)
        
        # 初始化
        self._reset_parameters()
        
    def _reset_parameters(self):
        for layer in [self.query, self.key, self.value, self.output_proj, self.gate]:
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
            
    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        计算多头注意力并聚合特征
        
        Args:
            x: 目标节点特征 [batch_size, in_channels]
            neighbors: 邻居节点特征 [batch_size, num_neighbors, in_channels]
            
        Returns:
            聚合后的特征 [batch_size, out_channels]
        """
        batch_size, num_neighbors, _ = neighbors.shape
        
        # 计算查询、键、值
        q = self.query(x).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.key(neighbors).view(batch_size, num_neighbors, self.num_heads, self.head_dim)
        v = self.value(neighbors).view(batch_size, num_neighbors, self.num_heads, self.head_dim)
        
        # 重排维度，将头维度放到前面
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, 1, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_neighbors, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_neighbors, head_dim]
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch_size, num_heads, 1, num_neighbors]
        
        # 加权求和
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
        
        # 重排维度，合并多头
        output = output.squeeze(2).permute(0, 2, 1)  # [batch_size, head_dim, num_heads]
        output = output.reshape(batch_size, -1)      # [batch_size, head_dim * num_heads]
        
        # 输出投影到指定的输出维度
        output = self.output_proj(output)  # [batch_size, out_channels]
        
        # 使用SE注意力增强特征
        output = self.se(output)
        
        # 门控机制 - 决定保留多少输入特征
        if x.size(1) != output.size(1):
            # 处理维度不匹配情况
            x_matched = nn.functional.pad(x, (0, output.size(1) - x.size(1)))
        else:
            x_matched = x
        
        gate_input = torch.cat([x, output], dim=1)
        gate_value = torch.sigmoid(self.gate(gate_input))
        
        output = gate_value * output + (1 - gate_value) * x_matched
        
        return output

class SAGEConv(nn.Module):
    """GraphSAGE卷积层，包含残差连接和归一化"""
    
    def __init__(self, in_channels: int, out_channels: int, use_layer_norm: bool = False, use_residual: bool = False):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # PyG的SAGEConv层
        self.conv = PYGSAGEConv(in_channels, out_channels, normalize=True)
        
        # 残差连接投影（如果输入输出维度不同）
        self.residual_proj = None
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
            
        # 层归一化
        self.layer_norm = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)
        
        # 添加GELU激活函数
        self.gelu = nn.GELU()
            
        # 初始化
        self._reset_parameters()
        
    def _reset_parameters(self):
        # PYGSAGEConv已经内置了初始化
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        # 使用PyG的SAGEConv
        out = self.conv(x, edge_index)
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + residual
            
        # 层归一化
        if self.layer_norm is not None:
            out = self.layer_norm(out)
            
        # 使用GELU激活函数
        out = self.gelu(out)
            
        return out

class FeedForward(nn.Module):
    """Feed-forward层，类似Transformer架构"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class GraphSAGE(nn.Module):
    """高性能GraphSAGE模型，支持多头注意力、残差连接和层归一化"""
    
    def __init__(self, in_channels: int, hidden_channels: List[int], 
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5,
                 use_layer_norm: bool = False, use_residual: bool = False,
                 attention_heads: int = 1):
        """
        初始化增强版GraphSAGE模型
        
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度列表
            out_channels: 输出特征维度
            num_layers: 层数
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
            use_residual: 是否使用残差连接
            attention_heads: 注意力头数
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.attention_heads = attention_heads
        
        # 确保隐藏层维度列表长度与层数相符
        if len(hidden_channels) < num_layers:
            # 用最后一个维度填充
            hidden_channels = hidden_channels + [hidden_channels[-1]] * (num_layers - len(hidden_channels))
        
        # 输入层
        self.input_lin = nn.Linear(in_channels, hidden_channels[0])
        
        # 中间层
        self.convs = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(
                hidden_channels[i], 
                hidden_channels[i+1],
                use_layer_norm=use_layer_norm,
                use_residual=use_residual
            ))
            # 修正：注意力层输入维度是当前层维度，输出维度是下一层维度
            self.attention_layers.append(AttentionLayer(
                hidden_channels[i],  # 输入维度：当前层维度
                hidden_channels[i+1],  # 输出维度：下一层维度，与卷积输出匹配
                num_heads=attention_heads
            ))
            # 添加Feed-forward网络
            self.feed_forwards.append(FeedForward(
                hidden_channels[i+1],
                hidden_channels[i+1] * 4,
                dropout=dropout
            ))
        
        # 输出层
        self.output_lin = nn.Linear(hidden_channels[-1], out_channels)
        
        # 批归一化层
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels[i]))
        
        # 初始化权重
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化模型参数"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            节点表示 [num_nodes, out_channels]
        """
        # 输入层
        x = self.input_lin(x)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 中间层
        for i in range(self.num_layers - 1):
            # 获取当前层的节点特征（用于注意力机制）
            current_x = x
            
            # GraphSAGE卷积
            x_sage = self.convs[i](x, edge_index)
            
            # 注意力聚合
            if i < len(self.attention_layers):
                # 获取邻居特征
                neighbors = self._get_neighbor_features(current_x, edge_index)
                
                # 注意力聚合
                x_att = self.attention_layers[i](current_x, neighbors)
                
                # 合并两种特征
                x = x_sage + F.dropout(x_att, p=self.dropout, training=self.training)
                
                # 应用Feed-forward网络
                x = x + F.dropout(self.feed_forwards[i](x), p=self.dropout, training=self.training)
                
                # 批归一化
                x = self.batch_norms[i+1](x)
            else:
                x = x_sage
                
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.output_lin(x)
        
        return x
    
    def _get_neighbor_features(self, x: torch.Tensor, edge_index: torch.Tensor, max_neighbors: int = 10) -> torch.Tensor:
        """
        获取邻居特征（优化版）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            max_neighbors: 每个节点的最大邻居数量
            
        Returns:
            邻居特征张量 [num_nodes, max_neighbors, hidden_channels]
        """
        num_nodes = x.size(0)
        feature_dim = x.size(1)
        device = x.device
        
        # 构建邻居列表（更高效）
        neighbor_lists = [[] for _ in range(num_nodes)]
        edge_index_cpu = edge_index.cpu()
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
            if len(neighbor_lists[src]) < max_neighbors:
                neighbor_lists[src].append(dst)
            if len(neighbor_lists[dst]) < max_neighbors:
                neighbor_lists[dst].append(src)
                
        # 填充邻居列表
        for i in range(num_nodes):
            if len(neighbor_lists[i]) == 0:
                # 自循环
                neighbor_lists[i] = [i] * max_neighbors
            elif len(neighbor_lists[i]) < max_neighbors:
                # 重复最后一个邻居
                last_neighbor = neighbor_lists[i][-1]
                padding = [last_neighbor] * (max_neighbors - len(neighbor_lists[i]))
                neighbor_lists[i].extend(padding)
            elif len(neighbor_lists[i]) > max_neighbors:
                # 截断
                neighbor_lists[i] = neighbor_lists[i][:max_neighbors]
                
        # 转换为张量并索引特征
        indices = torch.LongTensor(neighbor_lists).to(device)
        neighbors = x[indices]  # [num_nodes, max_neighbors, feature_dim]
        
        return neighbors

def train_model(model: GraphSAGE, data, epochs: int = 100, lr: float = 0.01, 
               weight_decay: float = 5e-4, patience: int = 10, return_metrics: bool = False,
               use_warmup: bool = False, warmup_epochs: int = 10, use_tqdm: bool = False):
    """
    训练模型
    
    Args:
        model: GraphSAGE模型
        data: 图数据
        epochs: 训练轮数
        lr: 学习率
        weight_decay: L2正则化系数
        patience: 早停耐心值
        return_metrics: 是否返回训练过程中的指标
        use_warmup: 是否使用预热
        warmup_epochs: 预热轮数
        use_tqdm: 是否使用进度条
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度策略
    if use_warmup:
        # 预热 + 余弦退火
        def warmup_cosine_schedule(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    else:
        # 常规减少学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 记录指标
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rate': []
    } if return_metrics else None
    
    # 检查数据
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask'):
        logger.warning("数据中缺少训练/验证掩码，创建默认掩码")
        num_nodes = data.x.size(0)
        indices = torch.randperm(num_nodes)
        
        # 80%训练, 10%验证, 10%测试
        train_size = int(0.8 * num_nodes)
        val_size = int(0.1 * num_nodes)
        
        # 创建掩码
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size+val_size]] = True
        data.test_mask[indices[train_size+val_size:]] = True
    
    # 检查目标变量
    if not hasattr(data, 'y') or data.y.shape[0] != data.x.size(0):
        logger.warning("数据中缺少有效目标变量，创建随机目标")
        data.y = torch.randint(0, 2, (data.x.size(0), 1), dtype=torch.float, device=device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 创建进度条
    epoch_iterator = range(epochs)
    if use_tqdm:
        epoch_iterator = tqdm(epoch_iterator, desc="训练进度")
    
    for epoch in epoch_iterator:
        # 训练模式
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data.x, data.edge_index)
        
        # 确保输出和目标尺寸一致
        if out.shape != data.y.shape:
            if out.shape[0] == data.y.shape[0]:
                # 尺寸调整
                if data.y.shape[1] == 1 and out.shape[1] > 1:
                    # 多类别输出到二分类
                    out = out[:, 0].unsqueeze(1)
                else:
                    # 其他调整
                    data.y = data.y.view(data.y.shape[0], -1)
                    out = out.view(out.shape[0], -1)
        
        # 计算损失
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # 反向传播
        train_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算训练指标
        with torch.no_grad():
            train_pred = (torch.sigmoid(out[data.train_mask]) > 0.5).float()
            train_acc = (train_pred == data.y[data.train_mask]).float().mean()
            
            # 计算F1分数
            train_f1 = f1_score(
                data.y[data.train_mask].cpu().numpy().flatten(),
                train_pred.cpu().numpy().flatten(),
                average='binary'
            )
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_pred = (torch.sigmoid(out[data.val_mask]) > 0.5).float()
            val_acc = (val_pred == data.y[data.val_mask]).float().mean()
            
            # 计算F1分数
            val_f1 = f1_score(
                data.y[data.val_mask].cpu().numpy().flatten(),
                val_pred.cpu().numpy().flatten(),
                average='binary'
            )
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if use_warmup:
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # 记录指标
        if return_metrics:
            metrics['train_loss'].append(train_loss.item())
            metrics['val_loss'].append(val_loss.item())
            metrics['train_acc'].append(train_acc.item())
            metrics['val_acc'].append(val_acc.item())
            metrics['train_f1'].append(train_f1)
            metrics['val_f1'].append(val_f1)
            metrics['learning_rate'].append(current_lr)
        
        # 更新进度条描述
        if use_tqdm:
            # 修复：直接设置新的postfix字典，而不是尝试合并
            postfix_dict = {
                'train_loss': f"{train_loss.item():.4f}",
                'val_loss': f"{val_loss.item():.4f}",
                'train_acc': f"{train_acc.item():.4f}",
                'val_acc': f"{val_acc.item():.4f}",
                'lr': f"{current_lr:.6f}"
            }
            epoch_iterator.set_postfix(**postfix_dict)
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                        f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
                        f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if use_tqdm:
                # 修复：更新带有"best"标记的进度条
                postfix_dict = {
                    'train_loss': f"{train_loss.item():.4f}",
                    'val_loss': f"{val_loss.item():.4f}",
                    'train_acc': f"{train_acc.item():.4f}",
                    'val_acc': f"{val_acc.item():.4f}",
                    'lr': f"{current_lr:.6f}",
                    'best': True
                }
                epoch_iterator.set_postfix(**postfix_dict)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"已恢复最佳模型（验证损失: {best_val_loss:.4f}）")
    
    return metrics if return_metrics else None


def evaluate_model(model: GraphSAGE, data, detailed: bool = False, use_tqdm: bool = False) -> Dict[str, Any]:
    """
    评估模型，并统计预测时间
    
    Args:
        model: GraphSAGE模型
        data: 图数据
        detailed: 是否返回详细指标
        use_tqdm: 是否使用进度条
        
    Returns:
        评估指标字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # 检查掩码和标签是否已存在
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
        logger.warning("评估前缺少必要的掩码，无法评估模型")
        return {
            'train_acc': 0.0,
            'val_acc': 0.0,
            'test_acc': 0.0,
            'prediction_time': 0.0
        }
        
    if not hasattr(data, 'y'):
        logger.warning("评估前缺少目标变量，无法评估模型")
        return {
            'train_acc': 0.0,
            'val_acc': 0.0,
            'test_acc': 0.0,
            'prediction_time': 0.0
        }
    
    model.eval()
    
    # 计时开始
    start_time = time.time()
    
    with torch.no_grad():
        # 前向传播
        out = model(data.x, data.edge_index)
        
        # 计时结束
        prediction_time = time.time() - start_time
        
        # 确保输出和目标尺寸一致
        if out.shape != data.y.shape:
            if out.shape[0] == data.y.shape[0]:
                # 需要调整输出尺寸
                data.y = data.y.view(data.y.shape[0], -1)
                out = out.view(out.shape[0], -1)
        
        # 转换为预测结果
        probs = torch.sigmoid(out)
        pred = (probs > 0.5).float()
        
        # 计算准确率
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        
        # 打印基本结果
        logger.info(f'训练准确率: {train_acc:.4f}')
        logger.info(f'验证准确率: {val_acc:.4f}')
        logger.info(f'测试准确率: {test_acc:.4f}')
        logger.info(f'预测总时间: {prediction_time:.4f}秒')
        
        if not detailed:
            return {
                'train_acc': train_acc.item(),
                'val_acc': val_acc.item(),
                'test_acc': test_acc.item(),
                'prediction_time': prediction_time
            }
        
        # 计算更详细的指标（带进度条）
        def calculate_metrics(mask, name):
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_prob = probs[mask].cpu().numpy()
            
            # 确保维度一致
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
            y_prob = y_prob.reshape(-1)
            
            acc = accuracy_score(y_true, y_pred)
            
            # 确保有正样本和负样本来计算其他指标
            if len(np.unique(y_true)) > 1:
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
            else:
                prec = rec = f1 = auc = 0.0
                
            return {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc
            }
        
        # 计算每个集合的详细指标
        logger.info("计算详细指标...")
        sets = ['训练集', '验证集', '测试集']
        masks = [data.train_mask, data.val_mask, data.test_mask]
        
        if use_tqdm:
            sets_iterator = tqdm(zip(sets, masks), total=3, desc="评估数据集")
        else:
            sets_iterator = zip(sets, masks)
            
        all_metrics = {}
        for name, mask in sets_iterator:
            metrics = calculate_metrics(mask, name)
            all_metrics[name.replace('集', '')] = metrics
        
        # 打印详细结果
        logger.info("\n详细评估指标:")
        for set_name, metrics in all_metrics.items():
            logger.info(f"{set_name}指标:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")
        
        return {
            'train': all_metrics['训练'],
            'val': all_metrics['验证'],
            'test': all_metrics['测试'],
            'prediction_time': prediction_time
        } 