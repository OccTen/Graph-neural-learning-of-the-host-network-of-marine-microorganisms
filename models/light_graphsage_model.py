#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级GraphSAGE模型
优化推理速度，减少复杂度但保持高性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
import logging
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightGraphSAGE(torch.nn.Module):
    """
    轻量级GraphSAGE模型
    - 减少隐藏层维度
    - 精简网络结构
    - 移除复杂的注意力机制
    - 保留批归一化和残差连接
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, dropout=0.2):
        super(LightGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入投影层 - 简化但保留
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        
        # 卷积层和批归一化
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 如果有更多层
        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 输出层 - 简单线性层
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
        self._init_weights()
        logging.info(f"初始化轻量级GraphSAGE: {num_layers}层, 隐藏维度={hidden_channels}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"模型参数数量: {total_params}")
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index):
        # 输入投影
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 保存输入特征用于残差连接
        h_prev = h
        
        # 应用图卷积层
        for i in range(self.num_layers):
            # 图卷积
            h_new = self.conv_layers[i](h, edge_index)
            
            # 批量归一化
            h_new = self.batch_norms[i](h_new)
            
            # 激活函数
            h_new = F.relu(h_new)
            
            # Dropout
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # 残差连接（每两层）
            if i % 2 == 1:
                h_new = h_new + h_prev
            
            # 更新特征和残差连接
            h_prev = h
            h = h_new
        
        # 输出投影
        output = self.output_layer(h)
        
        return output

def train_model(model, data, optimizer, criterion, scheduler=None, num_epochs=100, patience=10, lr_scheduler_type='plateau'):
    """
    轻量级训练函数
    - 简化训练逻辑
    - 保留早停和学习率调度
    - 移除混合精度支持以减少复杂性
    """
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
    
    # 检查数据是否平衡
    if hasattr(data, 'y') and data.y is not None:
        num_classes = data.y.max().item() + 1
        class_counts = [(data.y == i).sum().item() for i in range(num_classes)]
        class_weights = torch.tensor([1.0/c for c in class_counts], device=data.x.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"类别分布: {class_counts}, 使用加权损失")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # 确保训练掩码中有节点
        if not hasattr(data, 'train_mask') or data.train_mask.sum() == 0:
            data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
            
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        
        # 验证步骤
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
            else:
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
        
        # 定期打印
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, metrics

def evaluate_model(model, data):
    """轻量级评估函数"""
    model.eval()
    
    # 检查必要的属性是否存在
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask') or not hasattr(data, 'y'):
        logging.warning("缺少评估所需的属性")
        return {'accuracy': 0.0}, {'accuracy': 0.0}, {'accuracy': 0.0}
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # 计算各个集合的评估指标
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        
        # 简化的指标字典
        train_metrics = {'accuracy': train_acc}
        val_metrics = {'accuracy': val_acc}
        test_metrics = {'accuracy': test_acc}
        
        # 输出详细评估信息
        logging.info(f"训练集准确率: {train_acc:.4f}")
        logging.info(f"验证集准确率: {val_acc:.4f}")
        logging.info(f"测试集准确率: {test_acc:.4f}")
    
    return train_metrics, val_metrics, test_metrics

# 模型比较函数
def compare_model_size(light_model, full_model):
    """比较轻量模型与完整模型的大小和参数量"""
    light_params = sum(p.numel() for p in light_model.parameters())
    full_params = sum(p.numel() for p in full_model.parameters())
    
    reduction = (1 - light_params / full_params) * 100
    
    logging.info(f"轻量模型参数量: {light_params:,}")
    logging.info(f"完整模型参数量: {full_params:,}")
    logging.info(f"参数减少: {reduction:.2f}%")
    
    # 比较模型大小（序列化后）
    light_buffer = io.BytesIO()
    torch.save(light_model.state_dict(), light_buffer)
    light_size = light_buffer.getbuffer().nbytes
    
    full_buffer = io.BytesIO()
    torch.save(full_model.state_dict(), full_buffer)
    full_size = full_buffer.getbuffer().nbytes
    
    size_reduction = (1 - light_size / full_size) * 100
    
    logging.info(f"轻量模型大小: {light_size/1024:.2f} KB")
    logging.info(f"完整模型大小: {full_size/1024:.2f} KB")
    logging.info(f"大小减少: {size_reduction:.2f}%")
    
    return {
        'light_params': light_params,
        'full_params': full_params,
        'param_reduction': reduction,
        'light_size': light_size,
        'full_size': full_size,
        'size_reduction': size_reduction
    } 