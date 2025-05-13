#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化分析微生物与宿主之间的相互作用机制

使用方法：
python scripts/visualize_phage_host_interactions.py --input output/taxonomy_predictions.json --output output/interaction_analysis/
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from collections import defaultdict
import csv

# 设置matplotlib支持中文
import matplotlib
# 使用英文标签代替中文，避免字体问题
matplotlib.rcParams['axes.unicode_minus'] = False
# 检查系统类型并设置合适的字体
if sys.platform.startswith('win'):
    try:
        # 尝试使用微软雅黑字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    except:
        # 如果没有中文字体，则使用默认字体
        plt.rcParams['font.sans-serif'] = ['Arial']
else:
    # 非Windows系统
    plt.rcParams['font.sans-serif'] = ['Arial']

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化噬菌体-宿主互作关系')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件路径（taxonomy_predictions.json）')
    parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--taxonomy_data', type=str, help='分类学数据文件路径')
    parser.add_argument('--threshold', type=float, default=0.6, help='匹配阈值（默认0.6）')
    parser.add_argument('--top_n', type=int, default=20, help='展示前N个最重要的互作关系（默认20）')
    parser.add_argument('--interactive', action='store_true', help='生成交互式可视化')
    parser.add_argument('--fontsize', type=int, default=12, help='图表字体大小(默认12)')
    parser.add_argument('--dpi', type=int, default=300, help='图表DPI(默认300)')
    parser.add_argument('--english', action='store_true', help='使用英文标签代替中文', default=True)
    return parser.parse_args()

def load_data(input_path):
    """加载模型预测结果数据"""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        logging.info(f"成功加载数据: {input_path}")
        logging.info(f"数据包含 {len(data)} 个实体")
        return data
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        raise 

def extract_phage_host_data(data):
    """从预测结果中提取噬菌体和宿主数据"""
    phages = {}
    hosts = {}
    
    for entity_id, entity_data in data.items():
        if entity_data['type'] == 'phage':
            phages[entity_id] = entity_data
        elif entity_data['type'] == 'host':
            hosts[entity_id] = entity_data
    
    logging.info(f"提取到 {len(phages)} 个噬菌体和 {len(hosts)} 个宿主")
    return phages, hosts

def create_interaction_network(phages, hosts, threshold=0.6):
    """创建噬菌体-宿主互作网络"""
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点 - 噬菌体为一种颜色，宿主为另一种颜色
    for phage_id in phages:
        G.add_node(phage_id, type='phage', prediction=phages[phage_id]['prediction'])
    
    for host_id in hosts:
        G.add_node(host_id, type='host', prediction=hosts[host_id]['prediction'])
    
    # 计算并添加边 - 根据相似度建立联系
    edges = []
    for phage_id, phage in phages.items():
        for host_id, host in hosts.items():
            # 计算简单相似度 - 在实际应用中可以使用更复杂的相似度计算方法
            similarity = calculate_similarity(phage_id, phage, host_id, host)
            if similarity > threshold:
                edges.append((phage_id, host_id, {'weight': similarity}))
    
    # 添加边到网络
    G.add_edges_from(edges)
    
    logging.info(f"创建的网络包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    return G

def calculate_similarity(phage_id, phage, host_id, host):
    """计算噬菌体和宿主之间的相似度"""
    # 这是一个简化的相似度计算，实际应用中可以采用更复杂的方法
    # 例如基于序列比对、功能相似性等
    
    # 1. ID字符匹配
    id_similarity = 0
    for c1, c2 in zip(str(phage_id), str(host_id)):
        if c1 == c2:
            id_similarity += 1
            
    # 2. 检查子串匹配
    substr_match = 0
    if len(str(phage_id)) >= 3 and len(str(host_id)) >= 3:
        for i in range(len(str(phage_id))-2):
            substr = str(phage_id)[i:i+3]
            if substr in str(host_id):
                substr_match = 3
                break
    
    # 3. 考虑预测类别和置信度
    prediction_compatibility = 0
    # 如果噬菌体是类别0，宿主是类别1，增加兼容性分数
    if phage['prediction'] == 0 and host['prediction'] == 1:
        prediction_compatibility = 5
    
    # 计算总分 - 可以调整各部分权重
    similarity = id_similarity + substr_match + prediction_compatibility
    
    # 归一化到0-1范围
    return min(similarity / 15.0, 1.0)

def visualize_network(G, output_path, top_n=20, fontsize=12, dpi=300):
    """可视化互作网络"""
    plt.figure(figsize=(16, 12))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 提取最重要的互作关系（具有最高权重的边）
    if G.number_of_edges() > top_n:
        # 获取所有边的权重
        weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        # 按权重排序
        sorted_weights = sorted(weights, key=lambda x: x[2], reverse=True)
        # 获取前top_n个边
        top_edges = sorted_weights[:top_n]
        # 创建一个新图，只包含这些边和相应的节点
        H = nx.Graph()
        for u, v, w in top_edges:
            H.add_edge(u, v, weight=w)
            H.nodes[u]['type'] = G.nodes[u]['type']
            H.nodes[u]['prediction'] = G.nodes[u]['prediction']
            H.nodes[v]['type'] = G.nodes[v]['type']
            H.nodes[v]['prediction'] = G.nodes[v]['prediction']
        G = H
    
    # 节点位置 - 使用spring_layout获得更好的可视化效果
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # 节点颜色映射
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'phage':
            node_colors.append('skyblue')
        else:
            # 宿主按预测类别上色
            prediction = G.nodes[node]['prediction']
            if prediction == 0:
                node_colors.append('lightgreen')
            elif prediction == 1:
                node_colors.append('salmon')
            else:
                node_colors.append('purple')
    
    # 节点大小 - 根据连接数量调整
    node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]
    
    # 边宽度 - 根据权重调整
    edge_widths = [2 + 5 * G[u][v]['weight'] for u, v in G.edges()]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    
    # 绘制标签，限制数量以避免拥挤
    if len(G.nodes()) <= 30:
        nx.draw_networkx_labels(G, pos, font_size=fontsize, font_family='sans-serif')
    
    plt.title('Phage-Host Interaction Network', fontsize=fontsize+4)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"网络可视化已保存到: {output_path}")
    
    return G

def create_heatmap(phages, hosts, output_path, top_n=30, fontsize=12, dpi=300):
    """创建噬菌体-宿主互作热图"""
    # 选择前N个噬菌体和宿主
    selected_phages = list(phages.keys())[:min(top_n, len(phages))]
    selected_hosts = list(hosts.keys())[:min(top_n, len(hosts))]
    
    # 计算相似度矩阵
    similarity_matrix = np.zeros((len(selected_phages), len(selected_hosts)))
    
    for i, phage_id in enumerate(selected_phages):
        for j, host_id in enumerate(selected_hosts):
            similarity_matrix[i, j] = calculate_similarity(
                phage_id, phages[phage_id], 
                host_id, hosts[host_id]
            )
    
    # 创建热图
    plt.figure(figsize=(14, 10))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 自定义颜色映射，从白色到深蓝色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#0343df'])
    
    # 绘制热图
    ax = sns.heatmap(
        similarity_matrix, 
        cmap=cmap,
        annot=False,  # 不显示具体数值，避免拥挤
        linewidths=0.5,
        xticklabels=[str(h)[:10] + '...' for h in selected_hosts],  # 截断长ID
        yticklabels=[str(p)[:10] + '...' for p in selected_phages]
    )
    
    # 旋转x轴标签，使其更易读
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    
    plt.title('Phage-Host Interaction Heatmap', fontsize=fontsize+4)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"互作热图已保存到: {output_path}")

def visualize_embedding(phages, hosts, output_path, fontsize=12, dpi=300):
    """使用降维技术可视化噬菌体和宿主在特征空间中的分布"""
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 准备数据
    ids = []
    features = []
    types = []
    predictions = []
    
    # 转换特征
    for phage_id, phage in phages.items():
        try:
            if 'features' in phage:
                # 如果有原始特征，直接使用
                feature_vector = phage['features']
            elif 'probability' in phage:
                # 否则，使用预测概率作为特征
                feature_vector = phage['probability']
            else:
                # 如果没有合适的特征，使用一个默认值
                logging.warning(f"噬菌体 {phage_id} 缺少特征，使用默认值")
                feature_vector = [0, 0, 0]  # 默认向量
            
            ids.append(phage_id)
            features.append(feature_vector)
            types.append('phage')
            predictions.append(phage.get('prediction', 0))
        except Exception as e:
            logging.warning(f"处理噬菌体 {phage_id} 时出错: {str(e)}")
    
    for host_id, host in hosts.items():
        try:
            if 'features' in host:
                feature_vector = host['features']
            elif 'probability' in host:
                feature_vector = host['probability']
            else:
                logging.warning(f"宿主 {host_id} 缺少特征，使用默认值")
                feature_vector = [0, 0, 0]  # 默认向量
            
            ids.append(host_id)
            features.append(feature_vector)
            types.append('host')
            predictions.append(host.get('prediction', 1))
        except Exception as e:
            logging.warning(f"处理宿主 {host_id} 时出错: {str(e)}")
    
    # 确保至少有一些特征可以处理
    if not features:
        logging.error("没有有效的特征向量可供处理")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No valid feature data for visualization", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return
    
    # 确保所有特征向量长度一致
    try:
        lengths = [len(f) for f in features]
        min_length = min(lengths)
        if not all(len(f) == min_length for f in features):
            logging.warning(f"特征向量长度不一致，截断至最小长度: {min_length}")
            features = [f[:min_length] for f in features]
    except Exception as e:
        logging.error(f"处理特征向量长度时出错: {str(e)}")
        logging.debug(f"特征类型: {[type(f) for f in features]}")
        logging.debug(f"特征内容: {features}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error processing feature data: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return
    
    # 转换为numpy数组
    features_array = np.array(features)
    
    # 平衡数据集 - 对数量多的类别进行下采样
    # 先统计各类别数量
    phage_indices = [i for i, t in enumerate(types) if t == 'phage']
    host_class0_indices = [i for i, (t, p) in enumerate(zip(types, predictions)) if t == 'host' and p == 0]
    host_class1_indices = [i for i, (t, p) in enumerate(zip(types, predictions)) if t == 'host' and p == 1]
    host_special_indices = [i for i, (t, p) in enumerate(zip(types, predictions)) if t == 'host' and p > 1]
    
    # 如果噬菌体数量特别少，可以复制增加样本
    if len(phage_indices) < 10 and len(phage_indices) > 0:
        logging.info(f"噬菌体样本数量较少 ({len(phage_indices)}), 增加样本")
        # 复制噬菌体样本以增加平衡性
        multiplier = max(1, min(20 // len(phage_indices), 5))  # 增加样本但不超过5倍
        for _ in range(multiplier - 1):
            for idx in phage_indices:
                features_array = np.vstack([features_array, features_array[idx]])
                types.append('phage')
                predictions.append(predictions[idx])
                ids.append(ids[idx] + "_dup")
        
        # 更新索引
        phage_indices = [i for i, t in enumerate(types) if t == 'phage']
    
    # 使用PCA进行降维
    # 自适应选择PCA组件数量，不超过数据维度
    n_components = min(10, features_array.shape[0] - 1, features_array.shape[1])
    logging.info(f"使用PCA降维到 {n_components} 维")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_array)
    
    # 打印PCA解释方差比例
    explained_variance = pca.explained_variance_ratio_
    logging.info(f"PCA解释方差比例: {explained_variance}")
    logging.info(f"PCA累计解释方差: {np.sum(explained_variance):.4f}")
    
    # 使用t-SNE进一步降维到2D，优化参数
    # 适当增加perplexity以捕捉更全局的结构
    # 增加early_exaggeration以使聚类更明显
    # 增加迭代次数以获得更稳定的结果
    perplexity = min(40, max(5, len(features_array)//5))
    logging.info(f"使用t-SNE降维到2D，perplexity={perplexity}")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity,
        early_exaggeration=12.0,  # 默认是12，增大这个值可以使聚类更明显
        n_iter=2000,              # 默认是1000，增加迭代次数可以获得更稳定的结果
        n_iter_without_progress=300,  # 防止过早收敛
        min_grad_norm=1e-7,       # 更严格的收敛条件
        learning_rate='auto',      # 自动学习率
        metric='euclidean'        # 使用欧氏距离
    )
    tsne_result = tsne.fit_transform(pca_result)
    
    # 正规化t-SNE结果使其均匀分布在图上
    tsne_result = (tsne_result - tsne_result.min(axis=0)) / (tsne_result.max(axis=0) - tsne_result.min(axis=0))
    tsne_result = (tsne_result * 2 - 1) * 400  # 缩放到-400到400
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    
    # 为不同类型和预测创建颜色映射和标记
    colors = []
    markers = []
    edgecolors = []
    for i in range(len(types)):
        if types[i] == 'phage':
            colors.append('blue')
            markers.append('o')
            edgecolors.append('darkblue')
        else:
            # 宿主按预测类别上色
            if predictions[i] == 0:
                colors.append('green')
                markers.append('s')  # 方形
                edgecolors.append('darkgreen')
            elif predictions[i] == 1:
                colors.append('red')
                markers.append('o')  # 圆形
                edgecolors.append('darkred')
            else:
                colors.append('purple')
                markers.append('^')  # 三角形
                edgecolors.append('darkpurple')
    
    # 按类别分别绘制，确保所有类别都可见
    # 先画宿主，再画噬菌体，确保噬菌体在上层可见
    for category, label, marker in [
        ('host_1', 'Host (Class 1)', 'o'),
        ('host_0', 'Host (Class 0)', 's'),
        ('host_special', 'Special Host', '^'),
        ('phage', 'Phage', 'o')
    ]:
        indices = []
        if category == 'phage':
            indices = phage_indices
        elif category == 'host_0':
            indices = host_class0_indices
        elif category == 'host_1':
            indices = host_class1_indices
        elif category == 'host_special':
            indices = host_special_indices
        
        if not indices:
            continue
            
        # 获取该类别的颜色
        category_color = {
            'phage': 'blue',
            'host_0': 'green',
            'host_1': 'red',
            'host_special': 'purple'
        }[category]
        
        # 绘制该类别的点
        plt.scatter(
            tsne_result[indices, 0],
            tsne_result[indices, 1],
            c=category_color,
            marker=marker,
            s=80,  # 增大点的大小
            alpha=0.8,
            edgecolors='black',  # 添加黑色边框
            linewidths=0.5,
            label=label
        )
    
    # 添加图例
    plt.legend(fontsize=fontsize, loc='upper left')
    
    # 添加网格线以便更好地观察分布
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.title('Feature Space Distribution (t-SNE)', fontsize=fontsize+4)
    plt.xlabel('t-SNE Dimension 1', fontsize=fontsize+2)
    plt.ylabel('t-SNE Dimension 2', fontsize=fontsize+2)
    
    # 保持坐标轴等比例
    plt.axis('equal')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"特征空间可视化已保存到: {output_path}")
    
    # 输出类别统计信息
    logging.info(f"t-SNE可视化统计:")
    logging.info(f"- 噬菌体: {len(phage_indices)}")
    logging.info(f"- 宿主(类别0): {len(host_class0_indices)}")
    logging.info(f"- 宿主(类别1): {len(host_class1_indices)}")
    logging.info(f"- 特殊宿主: {len(host_special_indices)}")
    
    # 如果数据分布极其不平衡，输出警告
    if len(phage_indices) < 3 and max(len(host_class0_indices), len(host_class1_indices)) > 20:
        logging.warning("数据极度不平衡，噬菌体样本太少，可能影响可视化效果")

def analyze_taxonomy_patterns(phages, hosts, output_dir, fontsize=12, dpi=300):
    """分析分类学模式和互作关系"""
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 按预测类别分类
    predictions_by_type = defaultdict(lambda: defaultdict(int))
    for phage_id, phage in phages.items():
        predictions_by_type['phage'][phage['prediction']] += 1
    
    for host_id, host in hosts.items():
        predictions_by_type['host'][host['prediction']] += 1
    
    # 绘制条形图
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    types = list(predictions_by_type.keys())
    x = np.arange(len(types))
    width = 0.2
    
    # 找出所有可能的预测类别
    all_predictions = set()
    for type_pred in predictions_by_type.values():
        all_predictions.update(type_pred.keys())
    all_predictions = sorted(all_predictions)
    
    # 绘制每个预测类别的条形
    for i, pred in enumerate(all_predictions):
        counts = [predictions_by_type[t][pred] for t in types]
        plt.bar(x + i*width, counts, width, label=f'Class {pred}')
    
    plt.xlabel('Entity Type', fontsize=fontsize+2)
    plt.ylabel('Count', fontsize=fontsize+2)
    plt.title('Distribution by Type and Prediction Class', fontsize=fontsize+4)
    plt.xticks(x + width * (len(all_predictions) - 1) / 2, types, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图像
    taxonomy_path = os.path.join(output_dir, 'taxonomy_distribution.png')
    plt.savefig(taxonomy_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"分类学分布图已保存到: {taxonomy_path}")

def create_interactive_visualization(G, phages, hosts, output_path):
    """创建交互式网络可视化（使用HTML和JavaScript）"""
    try:
        # 检查是否安装了pyvis
        import pyvis
        from pyvis.network import Network
        
        # 创建网络
        net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
        
        # 从networkx复制数据
        for node, node_attrs in G.nodes(data=True):
            # 节点属性
            color = 'skyblue' if node_attrs['type'] == 'phage' else 'salmon'
            if node_attrs['type'] == 'host' and node_attrs['prediction'] != 1:
                color = 'purple' if node_attrs['prediction'] == 2 else 'lightgreen'
            
            # 添加节点信息
            title = f"ID: {node}<br>类型: {node_attrs['type']}<br>预测: {node_attrs['prediction']}"
            
            # 添加节点
            net.add_node(
                n_id=node, 
                label=str(node)[:15], 
                title=title,
                color=color, 
                size=25
            )
        
        # 添加边
        for source, target, edge_attrs in G.edges(data=True):
            width = 1 + 5 * edge_attrs['weight']
            net.add_edge(source, target, width=width, title=f"相似度: {edge_attrs['weight']:.3f}")
        
        # 设置物理布局选项
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001)
        
        # 保存到HTML文件
        net.save_graph(output_path)
        logging.info(f"交互式网络可视化已保存到: {output_path}")
        
    except ImportError:
        logging.warning("未安装pyvis库，无法创建交互式可视化。可以通过'pip install pyvis'安装。")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # 加载数据
    data = load_data(args.input)
    
    # 提取噬菌体和宿主数据
    phages, hosts = extract_phage_host_data(data)
    
    # 创建噬菌体-宿主互作网络
    G = create_interaction_network(phages, hosts, args.threshold)
    
    # 可视化互作网络
    network_path = os.path.join(args.output, 'interaction_network.png')
    visualize_network(G, network_path, args.top_n, args.fontsize, args.dpi)
    
    # 创建热图
    heatmap_path = os.path.join(args.output, 'interaction_heatmap.png')
    create_heatmap(phages, hosts, heatmap_path, 30, args.fontsize, args.dpi)
    
    # 可视化特征空间分布
    embedding_path = os.path.join(args.output, 'feature_embedding.png')
    visualize_embedding(phages, hosts, embedding_path, args.fontsize, args.dpi)
    
    # 分析分类学模式
    analyze_taxonomy_patterns(phages, hosts, args.output, args.fontsize, args.dpi)
    
    # 创建交互式可视化
    if args.interactive:
        interactive_path = os.path.join(args.output, 'interactive_network.html')
        create_interactive_visualization(G, phages, hosts, interactive_path)
    
    logging.info(f"所有可视化已完成并保存到 {args.output}")
    print(f"\n分析完成! 所有可视化图表已保存到目录: {args.output}")
    print("生成的可视化图表包括:")
    print(f"- 互作网络图: {network_path}")
    print(f"- 热图: {heatmap_path}")
    print(f"- 特征空间分布: {embedding_path}")
    print(f"- 分类学分布: {os.path.join(args.output, 'taxonomy_distribution.png')}")
    if args.interactive:
        print(f"- 交互式网络: {os.path.join(args.output, 'interactive_network.html')}")

if __name__ == "__main__":
    main() 