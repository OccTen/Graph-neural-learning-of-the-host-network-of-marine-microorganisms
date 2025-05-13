#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

def format_size(num_bytes):
    """将字节数格式化为人类可读的形式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def count_parameters(model):
    """计算模型的参数总数"""
    return sum(p.numel() for p in model.parameters())

def view_pt_file(model_path):
    """查看PT文件的内容和结构"""
    if not os.path.exists(model_path):
        print(f"错误：模型文件 '{model_path}' 不存在")
        return

    print(f"\n正在加载模型文件：{model_path}")
    print(f"文件大小：{format_size(os.path.getsize(model_path))}")

    try:
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("\n模型数据类型:", type(checkpoint))

        # 如果是字典，打印其keys
        if isinstance(checkpoint, dict):
            print("\n模型内容的键值:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], torch.Tensor):
                    print(f"  {key}: Tensor形状 {checkpoint[key].shape}, 数据类型 {checkpoint[key].dtype}")
                else:
                    print(f"  {key}: {type(checkpoint[key])}")
            
            # 分析model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("\n模型状态字典内容:")
                layers = {}
                total_model_params = 0
                
                for key, value in state_dict.items():
                    layer_name = key.split('.')[0] if '.' in key else key
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append(key)
                
                for layer_name, params in layers.items():
                    print(f"\n层 '{layer_name}':")
                    total_params = 0
                    for param in params:
                        if isinstance(state_dict[param], torch.Tensor):
                            shape = state_dict[param].shape
                            num_params = np.prod(shape)
                            total_params += num_params
                            total_model_params += num_params
                            print(f"  {param}: 形状 {shape}, 参数量 {num_params}")
                    print(f"  该层总参数量: {total_params:,}")
                
                print(f"\n模型总参数量: {total_model_params:,}")
            
            # 分析config
            if 'config' in checkpoint:
                print("\n模型配置:")
                config = checkpoint['config']
                if isinstance(config, dict):
                    for k, v in config.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  配置类型: {type(config)}")
            
            # 分析train_metrics
            if 'train_metrics' in checkpoint:
                print("\n训练指标:")
                metrics = checkpoint['train_metrics']
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  指标类型: {type(metrics)}")
            
            # 分析class_weights
            if 'class_weights' in checkpoint:
                weights = checkpoint['class_weights']
                print("\n类别权重:")
                print(f"  总类别数: {len(weights)}")
                print(f"  前5个类别权重: {weights[:5] if len(weights) >= 5 else weights}")
                print(f"  权重类型: {type(weights[0]) if weights else 'N/A'}")
                if len(weights) > 0:
                    print(f"  最小权重: {min(weights)}")
                    print(f"  最大权重: {max(weights)}")
                    print(f"  平均权重: {sum(weights) / len(weights)}")
            
            # 尝试打印一些特定信息
            if 'epoch' in checkpoint:
                print(f"\n模型训练轮次: {checkpoint['epoch']}")
            if 'best_val_acc' in checkpoint:
                print(f"最佳验证准确率: {checkpoint['best_val_acc']:.4f}")
            if 'args' in checkpoint:
                print("\n训练参数:")
                for k, v in vars(checkpoint['args']).items():
                    print(f"  {k}: {v}")
        
        # 如果是模型，打印模型结构
        elif isinstance(checkpoint, torch.nn.Module):
            print("\n模型结构:")
            print(checkpoint)
            print(f"\n模型总参数量: {count_parameters(checkpoint):,}")

        else:
            print("\n无法识别的模型格式")
    
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "weights/taxonomy_graphsage_optimized.pt"  # 默认路径
    
    view_pt_file(model_path) 