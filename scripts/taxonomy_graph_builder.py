#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类学图构建工具
- 从分类学名称构建图
- 创建节点特征
- 创建边索引
- 导出为标准格式
"""

import os
import pandas as pd
import numpy as np
import json
import torch
import logging
from collections import defaultdict
from pathlib import Path
import pickle
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'taxonomy_graph_builder.log'))
    ]
)

def find_file(filename, search_dirs=None):
    """查找文件在各种可能的位置"""
    if search_dirs is None:
        # 默认搜索位置
        search_dirs = [
            os.path.join('.', 'data'),
            os.path.abspath(os.path.join(os.getcwd(), 'data')),
            r"M:\4.9\data",
            r"M:\4.9"
        ]
    
    for directory in search_dirs:
        try:
            if os.path.exists(directory):
                path = os.path.join(directory, filename)
                if os.path.exists(path):
                    logging.info(f"找到文件: {path}")
                    return path
        except Exception as e:
            logging.warning(f"检查目录时出错 {directory}: {str(e)}")
    
    # 如果没有在指定目录找到，尝试在当前目录及其子目录中查找
    for root, dirs, files in os.walk('.'):
        if filename in files:
            path = os.path.join(root, filename)
            logging.info(f"在遍历中找到文件: {path}")
            return path
            
    logging.error(f"未能找到文件 {filename}")
    return None

def load_taxonomy_data(data_dir='data'):
    """加载分类学数据"""
    try:
        # 加载分类学值
        values_filename = 'taxonomy_values.json'
        values_path = os.path.join(data_dir, values_filename)
        
        if not os.path.exists(values_path):
            logging.warning(f"未在 {values_path} 找到分类学值文件，尝试其他位置")
            values_path = find_file(values_filename)
            
        if not values_path or not os.path.exists(values_path):
            logging.error(f"找不到分类学值文件: {values_filename}")
            return None, None
            
        with open(values_path, 'r') as f:
            taxonomy_values = json.load(f)
        
        # 加载分类学映射
        mapping_filename = 'taxonomy_node_mapping.json'
        mapping_path = os.path.join(data_dir, mapping_filename)
        
        if not os.path.exists(mapping_path):
            logging.warning(f"未在 {mapping_path} 找到分类学映射文件，尝试其他位置")
            mapping_path = find_file(mapping_filename)
        
        if not mapping_path or not os.path.exists(mapping_path):
            logging.error(f"找不到分类学映射文件: {mapping_filename}")
            return taxonomy_values, None
            
        with open(mapping_path, 'r') as f:
            taxonomy_mapping = json.load(f)
        
        logging.info(f"加载了分类学值，包含 {sum(len(values) for entity, levels in taxonomy_values.items() for level, values in levels.items())} 个值")
        logging.info(f"加载了分类学映射，包含 {len(taxonomy_mapping)} 个映射")
        
        return taxonomy_values, taxonomy_mapping
    except Exception as e:
        logging.error(f"加载分类学数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def create_node_features(taxonomy_values, taxonomy_mapping):
    """创建节点特征矩阵"""
    try:
        if not taxonomy_values or not taxonomy_mapping:
            logging.error("无法创建节点特征，分类学数据不完整")
            return None
        
        # 确定特征维度
        num_nodes = len(taxonomy_mapping)
        
        # 创建编码器字典 - 按分类级别对所有值进行独热编码
        encoders = {}
        feature_dims = {}
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        
        # 对于每个分类级别，收集所有值并创建编码器
        for level in taxonomy_levels:
            level_values = []
            for entity in taxonomy_values:
                if level in taxonomy_values[entity]:
                    level_values.extend(taxonomy_values[entity][level])
            
            # 确保值是唯一的
            level_values = list(set(level_values))
            
            # 创建编码器
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit([[val] for val in level_values])
            encoders[level] = encoder
            feature_dims[level] = len(level_values)
            
            logging.info(f"{level} 级别编码器: {len(level_values)} 个值")
        
        # 计算总特征维度
        total_dim = sum(feature_dims.values())
        logging.info(f"总特征维度: {total_dim}")
        
        # 创建特征矩阵
        features = np.zeros((num_nodes, total_dim))
        
        # 填充特征矩阵
        dim_offset = 0
        for level in taxonomy_levels:
            encoder = encoders[level]
            level_dim = feature_dims[level]
            
            # 遍历所有节点
            for node_key, node_id in taxonomy_mapping.items():
                # 解析节点键以获取实体、级别和值
                parts = node_key.split('_', 2)
                if len(parts) == 3:
                    entity_code, level_code, value = parts
                    
                    # 检查此节点是否属于当前级别
                    level_map = {'K': 'kingdom', 'P': 'phylum', 'C': 'class', 'O': 'order', 'F': 'family', 'G': 'genus'}
                    if level_code in level_map and level_map[level_code] == level:
                        # 编码值
                        try:
                            encoded = encoder.transform([[value]])
                            
                            # 将编码的特征添加到特征矩阵
                            features[node_id, dim_offset:dim_offset+level_dim] = encoded
                        except:
                            logging.warning(f"无法编码值: {value} (级别: {level})")
            
            # 更新维度偏移
            dim_offset += level_dim
        
        # 转换为张量
        node_features = torch.tensor(features, dtype=torch.float)
        logging.info(f"创建了节点特征矩阵，形状: {node_features.shape}")
        
        return node_features, encoders
    except Exception as e:
        logging.error(f"创建节点特征时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def load_taxonomy_relationships(file_path):
    """加载分类学关系数据"""
    try:
        if not os.path.exists(file_path):
            logging.error(f"找不到Excel文件: {file_path}")
            # 尝试其他位置
            alt_file = find_file('Training set.xlsx')
            if alt_file:
                file_path = alt_file
            else:
                return None
        
        # 读取第一个工作表
        df = pd.read_excel(file_path, sheet_name=0)
        logging.info(f"加载了Excel数据，形状: {df.shape}")
        
        # 检查第一行是否包含列名
        has_header = False
        if 'Phage' in df.columns and df.iloc[0]['Phage'] == 'accession':
            has_header = True
            logging.info("检测到第一行包含列名")
            df = df.iloc[1:].reset_index(drop=True)
        
        # 定义列名映射
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        entities = ['Phage', 'Host', 'Non-host']
        
        # 创建列名映射
        column_mapping = {}
        if has_header:
            # 第一行就是列名
            first_row = df.iloc[0]
            for i, col in enumerate(df.columns):
                for entity in entities:
                    if col == entity:
                        column_mapping[f"{entity.lower()}_accession"] = i
                        for j, level in enumerate(taxonomy_levels):
                            if i+j+1 < len(df.columns):
                                column_mapping[f"{entity.lower()}_{level}"] = i+j+1
        else:
            # 使用位置推断
            entity_indices = {}
            for entity in entities:
                try:
                    entity_indices[entity] = df.columns.get_loc(entity)
                except:
                    continue
            
            for entity, idx in entity_indices.items():
                column_mapping[f"{entity.lower()}_accession"] = idx
                for j, level in enumerate(taxonomy_levels):
                    if idx+j+1 < len(df.columns):
                        column_mapping[f"{entity.lower()}_{level}"] = idx+j+1
        
        logging.info(f"列名映射: {column_mapping}")
        
        # 创建关系列表
        relationships = []
        
        # 遍历每一行
        for i in range(len(df)):
            row_data = {}
            
            # 提取所有实体和分类级别的值
            for key, col_idx in column_mapping.items():
                row_data[key] = df.iloc[i, col_idx]
            
            # 添加到关系列表
            relationships.append(row_data)
        
        logging.info(f"提取了 {len(relationships)} 行关系数据")
        return relationships
    except Exception as e:
        logging.error(f"加载分类学关系数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_edge_index(relationships, taxonomy_mapping):
    """创建边索引"""
    try:
        if not relationships or not taxonomy_mapping:
            logging.error("无法创建边索引，关系数据或映射不完整")
            return None
        
        # 创建边源和目标列表
        edges_src = []
        edges_dst = []
        
        # 定义实体和分类级别
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        entity_codes = {'phage': 'P', 'host': 'H', 'non-host': 'N'}
        level_codes = {'kingdom': 'K', 'phylum': 'P', 'class': 'C', 'order': 'O', 'family': 'F', 'genus': 'G'}
        
        # 边类型计数
        edge_types = defaultdict(int)
        
        # 处理每一行关系
        for rel in relationships:
            # 对于每个实体，提取其所有分类级别的节点ID
            entity_nodes = {}
            for entity, code in entity_codes.items():
                entity_nodes[entity] = []
                
                for level, level_code in level_codes.items():
                    col_name = f"{entity}_{level}"
                    if col_name in rel and pd.notna(rel[col_name]):
                        # 创建节点键
                        node_key = f"{code}_{level_code}_{rel[col_name]}"
                        if node_key in taxonomy_mapping:
                            node_id = taxonomy_mapping[node_key]
                            entity_nodes[entity].append((level, node_id))
            
            # 创建实体内部的分类层次关系边
            for entity, nodes in entity_nodes.items():
                if len(nodes) > 1:
                    # 对于同一实体的不同分类级别，创建层次边
                    for i in range(len(nodes)-1):
                        level_i, node_i = nodes[i]
                        for j in range(i+1, len(nodes)):
                            level_j, node_j = nodes[j]
                            
                            # 添加双向边
                            edges_src.append(node_i)
                            edges_dst.append(node_j)
                            edge_types[f"hierarchy:{entity}:{level_i}-{level_j}"] += 1
                            
                            edges_src.append(node_j)
                            edges_dst.append(node_i)
                            edge_types[f"hierarchy:{entity}:{level_j}-{level_i}"] += 1
            
            # 创建实体间的关系边
            # Phage-Host关系
            if 'phage' in entity_nodes and 'host' in entity_nodes:
                for p_level, p_node in entity_nodes['phage']:
                    for h_level, h_node in entity_nodes['host']:
                        edges_src.append(p_node)
                        edges_dst.append(h_node)
                        edge_types[f"relation:phage-host:{p_level}-{h_level}"] += 1
                        
                        edges_src.append(h_node)
                        edges_dst.append(p_node)
                        edge_types[f"relation:host-phage:{h_level}-{p_level}"] += 1
            
            # Phage-NonHost关系
            if 'phage' in entity_nodes and 'non-host' in entity_nodes:
                for p_level, p_node in entity_nodes['phage']:
                    for n_level, n_node in entity_nodes['non-host']:
                        edges_src.append(p_node)
                        edges_dst.append(n_node)
                        edge_types[f"relation:phage-nonhost:{p_level}-{n_level}"] += 1
                        
                        edges_src.append(n_node)
                        edges_dst.append(p_node)
                        edge_types[f"relation:nonhost-phage:{n_level}-{p_level}"] += 1
        
        # 转换为张量
        if edges_src and edges_dst:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            logging.info(f"创建了边索引，形状: {edge_index.shape}")
            
            # 输出边类型统计
            logging.info("\n边类型统计:")
            for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  - {edge_type}: {count} 条边")
            
            return edge_index
        else:
            logging.error("未能创建任何边")
            return None
    except Exception as e:
        logging.error(f"创建边索引时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_node_labels(taxonomy_mapping):
    """创建节点标签"""
    try:
        if not taxonomy_mapping:
            logging.error("无法创建节点标签，映射数据不完整")
            return None
        
        # 解析节点键以创建标签
        labels = np.zeros(len(taxonomy_mapping))
        
        # 对实体类型进行标记
        entity_codes = {'P': 0, 'H': 1, 'N': 2}  # Phage, Host, Non-host
        
        for node_key, node_id in taxonomy_mapping.items():
            parts = node_key.split('_', 2)
            if len(parts) >= 1:
                entity_code = parts[0]
                if entity_code in entity_codes:
                    labels[node_id] = entity_codes[entity_code]
        
        # 转换为张量
        node_labels = torch.tensor(labels, dtype=torch.long)
        logging.info(f"创建了节点标签，形状: {node_labels.shape}")
        
        # 统计标签分布
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[int(label)] += 1
            
        logging.info("\n节点标签分布:")
        for label, count in sorted(label_counts.items()):
            label_name = {0: 'Phage', 1: 'Host', 2: 'Non-host'}.get(label, f'Unknown-{label}')
            logging.info(f"  - {label_name}: {count} 个节点")
        
        return node_labels
    except Exception as e:
        logging.error(f"创建节点标签时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def split_data(num_nodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据为训练、验证和测试集"""
    try:
        # 创建索引并随机排序
        indices = torch.randperm(num_nodes)
        
        # 计算划分大小
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        # 创建掩码
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # 填充掩码
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        logging.info(f"数据划分: 训练集 {train_mask.sum().item()} 节点, "
                     f"验证集 {val_mask.sum().item()} 节点, "
                     f"测试集 {test_mask.sum().item()} 节点")
        
        return train_mask, val_mask, test_mask
    except Exception as e:
        logging.error(f"划分数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None

def save_graph_data(node_features, edge_index, node_labels, train_mask, val_mask, test_mask, data_dir='data'):
    """保存图数据"""
    try:
        # 确保输出目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # 保存为PyTorch文件
        output_path = os.path.join(data_dir, 'taxonomy_graph_data.pt')
        torch.save(data, output_path)
        logging.info(f"图数据已保存至: {output_path}")
        
        # 同时保存单独的张量
        np.save(os.path.join(data_dir, 'node_features.npy'), node_features.numpy())
        np.save(os.path.join(data_dir, 'edge_index.npy'), edge_index.numpy())
        np.save(os.path.join(data_dir, 'node_labels.npy'), node_labels.numpy())
        np.save(os.path.join(data_dir, 'train_mask.npy'), train_mask.numpy())
        np.save(os.path.join(data_dir, 'val_mask.npy'), val_mask.numpy())
        np.save(os.path.join(data_dir, 'test_mask.npy'), test_mask.numpy())
        logging.info("已保存单独的张量文件")
        
        return output_path
    except Exception as e:
        logging.error(f"保存图数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def main():
    """主函数"""
    logging.info("开始构建分类学图...")
    
    # 确保data目录存在
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 列出当前目录和数据目录内容
    try:
        cwd = os.getcwd()
        logging.info(f"当前工作目录: {cwd}")
        contents = os.listdir(cwd)
        logging.info(f"目录内容: {contents}")
        
        # 查看data目录是否存在及其内容
        data_path = os.path.join(cwd, data_dir)
        if os.path.exists(data_path):
            data_contents = os.listdir(data_path)
            logging.info(f"data目录内容: {data_contents}")
    except Exception as e:
        logging.error(f"列出目录内容时出错: {str(e)}")
    
    # 加载分类学数据
    logging.info("正在加载分类学数据...")
    taxonomy_values, taxonomy_mapping = load_taxonomy_data(data_dir)
    
    if taxonomy_values is None:
        logging.error("无法加载分类学数据，退出")
        return
    
    # 如果没有映射数据，先运行分析脚本
    if taxonomy_mapping is None:
        logging.info("未找到分类学映射，请先运行taxonomy_analyzer.py脚本")
        return
    
    # 创建节点特征
    logging.info("正在创建节点特征...")
    node_features, encoders = create_node_features(taxonomy_values, taxonomy_mapping)
    
    if node_features is None:
        logging.error("无法创建节点特征，退出")
        return
    
    # 加载关系数据
    logging.info("正在加载分类学关系数据...")
    excel_file = 'Training set.xlsx'
    excel_path = find_file(excel_file)
    
    if not excel_path:
        logging.error(f"找不到Excel文件: {excel_file}")
        return
        
    relationships = load_taxonomy_relationships(excel_path)
    
    if relationships is None:
        logging.error("无法加载关系数据，退出")
        return
    
    # 创建边索引
    logging.info("正在创建边索引...")
    edge_index = create_edge_index(relationships, taxonomy_mapping)
    
    if edge_index is None:
        logging.error("无法创建边索引，退出")
        return
    
    # 创建节点标签
    logging.info("正在创建节点标签...")
    node_labels = create_node_labels(taxonomy_mapping)
    
    if node_labels is None:
        logging.error("无法创建节点标签，退出")
        return
    
    # 划分数据
    logging.info("正在划分数据...")
    train_mask, val_mask, test_mask = split_data(len(taxonomy_mapping))
    
    if train_mask is None:
        logging.error("无法划分数据，退出")
        return
    
    # 保存图数据
    logging.info("正在保存图数据...")
    output_path = save_graph_data(node_features, edge_index, node_labels, train_mask, val_mask, test_mask, data_dir)
    
    if output_path:
        logging.info(f"分类学图构建完成，已保存至: {output_path}")
    else:
        logging.error("保存图数据失败")

if __name__ == "__main__":
    main() 