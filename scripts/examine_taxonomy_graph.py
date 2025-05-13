#!/usr/bin/env python3
"""
检查分类学图数据结构
此脚本分析taxonomy_graph_data.pt文件，显示其结构和特征
"""
import os
import sys
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def find_taxonomy_graph_data():
    """查找分类学图数据文件"""
    data_paths = [
        os.path.join('data', 'taxonomy_graph_data.pt'),  # 相对路径
        os.path.join(current_dir, 'data', 'taxonomy_graph_data.pt'),  # 从当前目录
        os.path.join(os.path.dirname(current_dir), 'data', 'taxonomy_graph_data.pt'),  # 从父目录
        "M:\\4.9\\data\\taxonomy_graph_data.pt"  # 绝对路径
    ]
    
    for path in data_paths:
        logger.info(f"尝试从路径加载: {path}")
        if os.path.exists(path):
            try:
                data = torch.load(path)
                logger.info(f"成功从 {path} 加载数据")
                return data, path
            except Exception as e:
                logger.error(f"从 {path} 加载失败: {str(e)}")
    
    # 如果以上路径都失败，尝试查找文件
    logger.info("搜索数据文件...")
    for root, dirs, files in os.walk(os.path.dirname(current_dir)):
        if 'taxonomy_graph_data.pt' in files:
            path = os.path.join(root, 'taxonomy_graph_data.pt')
            try:
                data = torch.load(path)
                logger.info(f"成功从 {path} 加载数据")
                return data, path
            except Exception as e:
                logger.error(f"从 {path} 加载失败: {str(e)}")
    
    raise FileNotFoundError("找不到分类学图数据文件")

def find_node_mapping():
    """查找节点映射文件"""
    mapping_paths = [
        os.path.join('data', 'taxonomy_node_mapping.json'),
        os.path.join('data', 'node_mapping.json'),
        os.path.join(current_dir, 'data', 'taxonomy_node_mapping.json'),
        os.path.join(current_dir, 'data', 'node_mapping.json'),
        os.path.join(os.path.dirname(current_dir), 'data', 'taxonomy_node_mapping.json'),
        os.path.join(os.path.dirname(current_dir), 'data', 'node_mapping.json'),
        "M:\\4.9\\data\\taxonomy_node_mapping.json",
        "M:\\4.9\\data\\node_mapping.json"
    ]
    
    for path in mapping_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"成功从 {path} 加载节点映射")
                return mapping, path
            except Exception as e:
                logger.error(f"从 {path} 加载失败: {str(e)}")
    
    return None, None

def analyze_graph_structure(data):
    """分析图结构"""
    logger.info("=== 图结构分析 ===")
    edge_index = data.edge_index
    
    # 基本统计
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)
    logger.info(f"节点数量: {num_nodes}")
    logger.info(f"边数量: {num_edges}")
    logger.info(f"平均度: {num_edges / num_nodes:.2f}")
    
    # 度分布
    degrees = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        degrees[src] += 1
        if src != dst:  # 排除自环
            degrees[dst] += 1
    
    degree_counts = Counter(degrees.int().tolist())
    logger.info(f"最大度: {degrees.max().item()}")
    logger.info(f"最小度: {degrees.min().item()}")
    logger.info(f"中位数度: {torch.median(degrees).item()}")
    
    # 自环数量
    self_loops = sum(1 for i in range(edge_index.size(1)) if edge_index[0, i] == edge_index[1, i])
    logger.info(f"自环数量: {self_loops}")
    
    # 连通分量
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        neighbors = [edge_index[1, i].item() for i in range(edge_index.size(1)) if edge_index[0, i].item() == node]
        for neighbor in neighbors:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in range(num_nodes):
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    logger.info(f"连通分量数量: {len(components)}")
    logger.info(f"最大连通分量大小: {max(len(c) for c in components)}")
    
    return degrees, components

def analyze_node_features(data, node_mapping=None):
    """分析节点特征"""
    logger.info("=== 节点特征分析 ===")
    
    x = data.x
    logger.info(f"特征维度: {x.size(1)}")
    
    # 特征统计
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0)
    feature_min = x.min(dim=0)[0]
    feature_max = x.max(dim=0)[0]
    
    logger.info(f"特征均值范围: [{feature_mean.min().item():.4f}, {feature_mean.max().item():.4f}]")
    logger.info(f"特征标准差范围: [{feature_std.min().item():.4f}, {feature_std.max().item():.4f}]")
    logger.info(f"特征极值范围: [{feature_min.min().item():.4f}, {feature_max.max().item():.4f}]")
    
    # 非零特征比例
    nonzero_ratio = (x != 0).float().mean().item()
    logger.info(f"非零特征比例: {nonzero_ratio:.4f}")
    
    # 特征有效性分析
    zero_features = ((x == 0).sum(dim=0) == x.size(0)).sum().item()
    const_features = ((x == x[0]).all(dim=0)).sum().item()
    logger.info(f"全零特征数量: {zero_features}")
    logger.info(f"常数特征数量: {const_features}")
    
    # 显示前10个节点的前10个特征值
    if node_mapping and len(node_mapping) > 0:
        logger.info("节点映射示例:")
        sample_count = min(5, len(node_mapping))
        sample_nodes = list(node_mapping.keys())[:sample_count]
        
        for node_name in sample_nodes:
            node_id = node_mapping[node_name]
            if isinstance(node_id, list):
                node_id = node_id[0]  # 如果是列表，取第一个
            
            if 0 <= node_id < x.size(0):
                logger.info(f"节点 '{node_name}' (ID: {node_id}):")
                logger.info(f"  前10个特征: {x[node_id, :10].tolist()}")
            else:
                logger.info(f"节点 '{node_name}' ID超出范围: {node_id}")
    else:
        logger.info("前5个节点的前10个特征值:")
        for i in range(min(5, x.size(0))):
            logger.info(f"节点 {i}: {x[i, :10].tolist()}")
    
    return x

def analyze_labels(data):
    """分析标签"""
    logger.info("=== 标签分析 ===")
    
    if not hasattr(data, 'y') or data.y is None:
        logger.info("数据中没有标签")
        return None
    
    y = data.y
    
    # 标签基本信息
    logger.info(f"标签形状: {y.shape}")
    
    if y.dim() == 1 or y.size(1) == 1:
        # 单标签分类
        if y.dim() == 2:
            y = y.squeeze(1)
        
        label_counts = Counter(y.tolist())
        logger.info(f"标签类型: 单标签分类")
        logger.info(f"类别数量: {len(label_counts)}")
        logger.info(f"类别分布: {dict(label_counts)}")
        
        # 检查标签与掩码的分布
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            train_label_counts = Counter(y[data.train_mask].tolist())
            val_label_counts = Counter(y[data.val_mask].tolist()) if hasattr(data, 'val_mask') else {}
            test_label_counts = Counter(y[data.test_mask].tolist()) if hasattr(data, 'test_mask') else {}
            
            logger.info(f"训练集类别分布: {dict(train_label_counts)}")
            if val_label_counts:
                logger.info(f"验证集类别分布: {dict(val_label_counts)}")
            if test_label_counts:
                logger.info(f"测试集类别分布: {dict(test_label_counts)}")
    else:
        # 多标签分类
        logger.info(f"标签类型: 多标签分类")
        logger.info(f"类别数量: {y.size(1)}")
        
        # 计算每个类别的样本数
        class_counts = y.sum(dim=0).tolist()
        class_dist = {i: count for i, count in enumerate(class_counts)}
        logger.info(f"类别分布: {class_dist}")
        
        # 计算每个节点的标签数量
        labels_per_node = y.sum(dim=1)
        avg_labels = labels_per_node.float().mean().item()
        logger.info(f"平均每个节点的标签数量: {avg_labels:.2f}")
    
    return y

def plot_degree_distribution(degrees):
    """绘制度分布"""
    try:
        degree_counts = Counter(degrees.int().tolist())
        degrees_list = list(degree_counts.keys())
        counts = list(degree_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(degrees_list, counts, width=0.8)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('节点度')
        plt.ylabel('节点数量')
        plt.title('节点度分布（双对数坐标）')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(os.path.dirname(current_dir), 'plots')
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, 'degree_distribution.png')
        plt.savefig(save_path)
        logger.info(f"度分布图已保存到: {save_path}")
        plt.close()
    except Exception as e:
        logger.error(f"绘制度分布时出错: {str(e)}")

def plot_feature_visualization(x, y=None, method='pca'):
    """使用降维技术可视化特征"""
    try:
        n_samples = min(1000, x.size(0))  # 最多使用1000个样本
        indices = torch.randperm(x.size(0))[:n_samples]
        
        X_sample = x[indices].numpy()
        
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
        
        X_2d = reducer.fit_transform(X_sample)
        
        plt.figure(figsize=(10, 8))
        
        if y is not None and (y.dim() == 1 or y.size(1) == 1):
            if y.dim() == 2:
                y = y.squeeze(1)
            
            y_sample = y[indices].numpy()
            unique_labels = np.unique(y_sample)
            
            for label in unique_labels:
                mask = y_sample == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'类别 {label}', alpha=0.7)
            
            plt.legend()
        else:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
        
        plt.title(f'特征可视化 ({method.upper()})')
        plt.xlabel('维度 1')
        plt.ylabel('维度 2')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(os.path.dirname(current_dir), 'plots')
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, f'feature_visualization_{method}.png')
        plt.savefig(save_path)
        logger.info(f"特征可视化已保存到: {save_path}")
        plt.close()
    except Exception as e:
        logger.error(f"绘制特征可视化时出错: {str(e)}")

def main():
    """主函数"""
    logger.info("开始分析分类学图数据...")
    
    try:
        # 1. 加载数据
        data, data_path = find_taxonomy_graph_data()
        logger.info(f"成功加载数据: {data_path}")
        
        # 2. 加载节点映射（如果存在）
        node_mapping, mapping_path = find_node_mapping()
        if node_mapping:
            logger.info(f"成功加载节点映射: {mapping_path} (包含 {len(node_mapping)} 个映射)")
        else:
            logger.info("未找到节点映射文件")
        
        # 3. 分析图结构
        degrees, components = analyze_graph_structure(data)
        
        # 4. 分析节点特征
        features = analyze_node_features(data, node_mapping)
        
        # 5. 分析标签
        labels = analyze_labels(data)
        
        # 6. 可视化
        if has_matplotlib():
            plot_degree_distribution(degrees)
            if features is not None:
                plot_feature_visualization(features, labels, method='pca')
                if features.size(0) <= 1000:  # t-SNE对大数据集可能会很慢
                    plot_feature_visualization(features, labels, method='tsne')
        
        # 7. 保存分析结果
        save_analysis_results(data, degrees, components, node_mapping)
        
        logger.info("分析完成!")
        return True
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def has_matplotlib():
    """检查是否可以使用matplotlib"""
    try:
        import matplotlib
        return True
    except ImportError:
        logger.warning("未安装matplotlib，跳过可视化步骤")
        return False

def save_analysis_results(data, degrees, components, node_mapping):
    """保存分析结果到JSON文件"""
    analysis_result = {
        "graph_stats": {
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.size(1),
            "avg_degree": float(degrees.mean().item()),
            "max_degree": int(degrees.max().item()),
            "min_degree": int(degrees.min().item()),
            "num_components": len(components),
            "largest_component_size": max(len(c) for c in components),
        },
        "feature_stats": {
            "feature_dim": data.x.size(1),
            "feature_mean": float(data.x.mean().item()),
            "feature_std": float(data.x.std().item()),
            "nonzero_ratio": float((data.x != 0).float().mean().item()),
        }
    }
    
    if hasattr(data, 'y') and data.y is not None:
        y = data.y
        if y.dim() == 1 or y.size(1) == 1:
            if y.dim() == 2:
                y = y.squeeze(1)
            label_counts = Counter(y.tolist())
            analysis_result["label_stats"] = {
                "num_classes": len(label_counts),
                "class_distribution": {str(k): v for k, v in label_counts.items()}
            }
        else:
            class_counts = y.sum(dim=0).tolist()
            analysis_result["label_stats"] = {
                "num_classes": y.size(1),
                "class_distribution": {str(i): count for i, count in enumerate(class_counts)},
                "avg_labels_per_node": float(y.sum(dim=1).float().mean().item())
            }
    
    if node_mapping:
        # 只保存前10个映射作为示例
        sample_mapping = dict(list(node_mapping.items())[:10])
        analysis_result["node_mapping_sample"] = sample_mapping
    
    # 保存到文件
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'taxonomy_graph_analysis.json')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"分析结果已保存到: {save_path}")

if __name__ == "__main__":
    main() 