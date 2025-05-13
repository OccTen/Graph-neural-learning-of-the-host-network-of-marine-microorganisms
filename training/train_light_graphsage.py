#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级GraphSAGE模型训练脚本
测试轻量模型性能与复杂模型对比
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import logging
import gc
import time
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.light_graphsage_model import LightGraphSAGE, train_model, evaluate_model, compare_model_size
from models.best_graphsage_model import GraphSAGE
from training.train_best_graphsage import enhance_graph_data, generate_meaningful_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_compare_metrics(light_metrics, full_metrics, save_path=None):
    """比较轻量模型和完整模型的训练指标"""
    plt.figure(figsize=(15, 10))
    
    # 绘制训练损失比较
    plt.subplot(2, 2, 1)
    plt.plot(light_metrics['train_loss'], label='轻量模型')
    plt.plot(full_metrics['train_loss'], label='完整模型')
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制验证损失比较
    plt.subplot(2, 2, 2)
    plt.plot(light_metrics['val_loss'], label='轻量模型')
    plt.plot(full_metrics['val_loss'], label='完整模型')
    plt.title('验证损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制训练准确率比较
    plt.subplot(2, 2, 3)
    plt.plot(light_metrics['train_acc'], label='轻量模型')
    plt.plot(full_metrics['train_acc'], label='完整模型')
    plt.title('训练准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制验证准确率比较
    plt.subplot(2, 2, 4)
    plt.plot(light_metrics['val_acc'], label='轻量模型')
    plt.plot(full_metrics['val_acc'], label='完整模型')
    plt.title('验证准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"比较结果已保存至 {save_path}")
    plt.show()

def benchmark_inference(light_model, full_model, data, num_runs=100, batch_size=None):
    """对比两个模型的推理性能"""
    device = next(light_model.parameters()).device
    
    # 准备测试数据
    if batch_size is None:
        # 全图推理
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # 预热
        with torch.no_grad():
            _ = light_model(x, edge_index)
            _ = full_model(x, edge_index)
        
        # 轻量模型计时
        light_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                _ = light_model(x, edge_index)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            light_times.append(time.time() - start)
        
        # 完整模型计时
        full_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                _ = full_model(x, edge_index)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            full_times.append(time.time() - start)
    else:
        # 批处理推理
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        # 轻量模型计时
        light_times = []
        for _ in range(num_runs):
            batch_idx = indices[:batch_size]
            batch_mask = torch.zeros(num_nodes, dtype=torch.bool)
            batch_mask[batch_idx] = True
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                out = light_model(data.x, data.edge_index)
                _ = out[batch_mask]
            torch.cuda.synchronize() if device.type == 'cuda' else None
            light_times.append(time.time() - start)
        
        # 完整模型计时
        full_times = []
        for _ in range(num_runs):
            batch_idx = indices[:batch_size]
            batch_mask = torch.zeros(num_nodes, dtype=torch.bool)
            batch_mask[batch_idx] = True
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                out = full_model(data.x, data.edge_index)
                _ = out[batch_mask]
            torch.cuda.synchronize() if device.type == 'cuda' else None
            full_times.append(time.time() - start)
    
    # 计算统计结果
    light_avg = np.mean(light_times) * 1000  # 转为毫秒
    light_std = np.std(light_times) * 1000
    full_avg = np.mean(full_times) * 1000
    full_std = np.std(full_times) * 1000
    
    speedup = full_avg / light_avg
    
    logging.info(f"轻量模型平均推理时间: {light_avg:.4f} ± {light_std:.4f} ms")
    logging.info(f"完整模型平均推理时间: {full_avg:.4f} ± {full_std:.4f} ms")
    logging.info(f"速度提升: {speedup:.2f}x")
    
    return {
        'light_avg': light_avg,
        'light_std': light_std,
        'full_avg': full_avg,
        'full_std': full_std,
        'speedup': speedup
    }

def main():
    """主函数"""
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
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    
    data = torch.load(data_path)
    logging.info(f"成功加载数据: {data.num_nodes}节点, {data.edge_index.size(1)}边")
    
    # 增强图数据
    data = enhance_graph_data(data, augment=True)
    
    # 生成标签
    data.y = generate_meaningful_labels(data, strategy='complex')
    
    # 创建掩码
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 将数据移动到设备
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    
    # 轻量模型配置
    light_hidden_dim = 64
    light_num_layers = 2
    light_dropout = 0.2
    
    # 加载完整模型作为对比
    full_model_path = os.path.join(root_dir, 'weights', 'best_graphsage_model.pt')
    if os.path.exists(full_model_path):
        full_model = GraphSAGE(
            in_channels=data.x.size(1),
            hidden_channels=128,
            out_channels=2,
            num_layers=3,
            dropout=0.3
        ).to(device)
        full_model.load_state_dict(torch.load(full_model_path, map_location=device))
        logging.info(f"加载完整模型权重: {full_model_path}")
        load_full_model = True
    else:
        logging.warning(f"完整模型权重未找到: {full_model_path}")
        full_model = GraphSAGE(
            in_channels=data.x.size(1),
            hidden_channels=128,
            out_channels=2,
            num_layers=3,
            dropout=0.3
        ).to(device)
        load_full_model = False
    
    # 初始化轻量模型
    light_model = LightGraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=light_hidden_dim,
        out_channels=2,
        num_layers=light_num_layers,
        dropout=light_dropout
    ).to(device)
    
    # 比较模型大小和参数量
    size_comparison = compare_model_size(light_model, full_model)
    
    # 如果有预训练的完整模型，评估性能并进行推理基准测试
    if load_full_model:
        logging.info("评估完整模型性能...")
        try:
            full_train_metrics, full_val_metrics, full_test_metrics = evaluate_model(full_model, data)
            
            logging.info("进行推理性能基准测试...")
            inference_benchmark = benchmark_inference(light_model, full_model, data, num_runs=50)
        except Exception as e:
            logging.error(f"评估完整模型或基准测试时出错: {e}")
            # 设置默认值，避免后续引用错误
            full_train_metrics = {'accuracy': 0.0}
            full_val_metrics = {'accuracy': 0.0}
            full_test_metrics = {'accuracy': 0.0}
            inference_benchmark = {
                'light_avg': 0.0,
                'light_std': 0.0,
                'full_avg': 0.0,
                'full_std': 0.0,
                'speedup': 1.0
            }
            load_full_model = False  # 降级处理
    
    # 训练轻量模型
    learning_rate = 0.005
    weight_decay = 5e-4
    num_epochs = 100
    patience = 15
    
    optimizer = AdamW(light_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    logging.info("开始训练轻量模型...")
    train_start = time.time()
    light_model, light_metrics = train_model(
        light_model, 
        data, 
        optimizer, 
        criterion, 
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=patience,
        lr_scheduler_type='cosine'
    )
    train_end = time.time()
    training_time = train_end - train_start
    logging.info(f"轻量模型训练完成，耗时: {training_time:.2f}秒")
    
    # 评估轻量模型
    logging.info("评估轻量模型性能...")
    light_train_metrics, light_val_metrics, light_test_metrics = evaluate_model(light_model, data)
    
    # 保存轻量模型
    weights_dir = os.path.join(root_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    light_model_path = os.path.join(weights_dir, 'light_graphsage_model.pt')
    torch.save(light_model.state_dict(), light_model_path)
    logging.info(f"轻量模型保存至: {light_model_path}")
    
    # 如果有预训练的完整模型，比较训练指标
    if load_full_model:
        # 加载完整模型训练指标
        try:
            full_metrics_path = os.path.join(root_dir, 'results', 'full_model_metrics.pt')
            if os.path.exists(full_metrics_path):
                full_metrics = torch.load(full_metrics_path)
                
                # 绘制对比图
                results_dir = os.path.join(root_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                compare_path = os.path.join(results_dir, 'model_comparison.png')
                plot_compare_metrics(light_metrics, full_metrics, save_path=compare_path)
            else:
                logging.warning(f"完整模型训练指标文件未找到: {full_metrics_path}")
        except Exception as e:
            logging.error(f"加载完整模型训练指标失败: {e}")
    
    # 保存轻量模型的训练指标
    metrics_path = os.path.join(root_dir, 'results', 'light_model_metrics.pt')
    torch.save(light_metrics, metrics_path)
    
    # 保存评估结果到CSV
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'model_comparison.csv')
    
    try:
        with open(results_path, 'w') as f:
            f.write("metric,light_model,full_model\n")
            f.write(f"train_accuracy,{light_train_metrics['accuracy']:.4f},{full_train_metrics['accuracy'] if load_full_model else 'N/A'}\n")
            f.write(f"val_accuracy,{light_val_metrics['accuracy']:.4f},{full_val_metrics['accuracy'] if load_full_model else 'N/A'}\n")
            f.write(f"test_accuracy,{light_test_metrics['accuracy']:.4f},{full_test_metrics['accuracy'] if load_full_model else 'N/A'}\n")
            f.write(f"parameters,{size_comparison['light_params']},{size_comparison['full_params']}\n")
            f.write(f"model_size_kb,{size_comparison['light_size']/1024:.2f},{size_comparison['full_size']/1024:.2f}\n")
            
            if load_full_model:
                f.write(f"inference_time_ms,{inference_benchmark['light_avg']:.4f},{inference_benchmark['full_avg']:.4f}\n")
                f.write(f"speedup,{inference_benchmark['speedup']:.2f},1.00\n")
            
            f.write(f"training_time_s,{training_time:.2f},N/A\n")
        
        logging.info(f"比较结果已保存至: {results_path}")
    except Exception as e:
        logging.error(f"保存比较结果失败: {e}")
    
    # 更新模型版本文档
    try:
        model_versions_path = os.path.join(root_dir, 'weights', 'model_versions.md')
        with open(model_versions_path, 'a') as f:
            f.write("\n\n## 轻量级模型（新增）\n")
            f.write("- 文件名：light_graphsage_model.py\n")
            f.write("- 位置：models/\n")
            f.write("- 特点：\n")
            f.write(f"  - {light_num_layers}层GraphSAGE结构\n")
            f.write(f"  - 隐藏层维度{light_hidden_dim}\n")
            f.write("  - 移除注意力机制\n")
            f.write("  - 保留批归一化和残差连接\n")
            f.write(f"  - dropout率{light_dropout}\n")
            f.write(f"  - 参数减少{size_comparison['param_reduction']:.2f}%\n")
            if load_full_model:
                f.write(f"  - 推理速度提升{inference_benchmark['speedup']:.2f}x\n")
            f.write(f"  - 测试准确率：{light_test_metrics['accuracy']:.4f}\n")
            f.write(f"  - **训练日期**: {time.strftime('%Y-%m-%d')}\n")
        logging.info(f"模型版本文档已更新: {model_versions_path}")
    except Exception as e:
        logging.error(f"更新模型版本文档失败: {e}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"结束时GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

if __name__ == "__main__":
    main() 