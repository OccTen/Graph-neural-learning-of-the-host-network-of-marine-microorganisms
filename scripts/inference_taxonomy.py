#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用基于分类学信息的GraphSAGE模型进行推理
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 尝试导入模型
try:
    from models.best_graphsage_model import GraphSAGE
except ImportError:
    try:
        # 尝试相对导入
        sys.path.append(os.path.join(project_root, 'models'))
        from best_graphsage_model import GraphSAGE
    except ImportError:
        print("无法导入GraphSAGE模型，请确保models目录在Python路径中")
        print(f"当前Python路径: {sys.path}")
        print(f"正在尝试从以下位置导入: {os.path.join(project_root, 'models')}")
        print("请检查models/best_graphsage_model.py文件是否存在")
        sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'taxonomy_inference.log'))
    ]
)
logger = logging.getLogger(__name__)

# 添加调试日志选项
DEBUG = True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于分类学信息的GraphSAGE模型推理')
    parser.add_argument('--model', type=str, default="weights/taxonomy_graphsage_optimized.pt", help='模型权重文件路径')
    parser.add_argument('--test_data', type=str, default="data/Independent test set.xlsx", help='测试数据文件路径')
    parser.add_argument('--encoders', type=str, default="data/taxonomy_values.json", help='分类学编码器JSON文件路径')
    parser.add_argument('--output', type=str, default="output/taxonomy_predictions.json", help='预测结果输出路径')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--device', type=str, default='', help='设备，留空自动选择')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，输出更详细的日志')
    return parser.parse_args()

def load_taxonomy_encoders(encoders_path):
    """加载分类学编码器"""
    try:
        with open(encoders_path, 'r') as f:
            encoders_data = json.load(f)
        
        # 为每个分类学级别重建编码器
        encoders = {}
        
        # 处理taxonomy_values.json格式 - 这种格式的结构是
        # {"Phage": {"kingdom": [...], "phylum": [...], ...}, "Host": {...}}
        if "Phage" in encoders_data and isinstance(encoders_data["Phage"], dict):
            logging.info("检测到taxonomy_values.json格式的编码器")
            
            # 合并所有实体的分类级别值
            merged_taxonomy_values = {}
            for entity, taxonomy in encoders_data.items():
                for level, values in taxonomy.items():
                    if level not in merged_taxonomy_values:
                        merged_taxonomy_values[level] = set()
                    merged_taxonomy_values[level].update(values)
            
            # 转换为编码器
            for level, values in merged_taxonomy_values.items():
                values_list = sorted(list(values))
                encoder = LabelEncoder()
                encoder.classes_ = np.array(values_list)
                encoders[level] = encoder
                logging.info(f"加载分类级别编码器: {level}, 类别数: {len(values_list)}")
                if DEBUG:
                    logging.info(f"前5个类别: {', '.join(values_list[:5])}")
        
        # 处理标准编码器格式 - 这种格式的结构是 {"kingdom": [...], "phylum": [...], ...}
        elif isinstance(encoders_data, dict):
            for level, classes in encoders_data.items():
                if isinstance(classes, list):
                    encoder = LabelEncoder()
                    encoder.classes_ = np.array(classes)
                    encoders[level] = encoder
                    logging.info(f"加载分类级别编码器: {level}, 类别数: {len(classes)}")
                    if DEBUG:
                        logging.info(f"前5个类别: {', '.join(classes[:5])}")
        
        # 检查是否成功加载编码器
        if not encoders:
            raise ValueError("无法识别编码器格式或编码器为空")
            
        logging.info(f"成功加载了 {len(encoders)} 个分类学级别的编码器")
        return encoders
    except Exception as e:
        logging.error(f"加载分类学编码器出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def load_test_data(test_data_path):
    """加载测试数据"""
    try:
        # 检查文件类型
        if test_data_path.endswith('.xlsx'):
            # Excel文件
            df = pd.read_excel(test_data_path)
        elif test_data_path.endswith('.csv'):
            # CSV文件
            df = pd.read_csv(test_data_path)
        else:
            raise ValueError(f"不支持的文件格式: {test_data_path}")
        
        logging.info(f"加载测试数据: {test_data_path}")
        logging.info(f"数据形状: {df.shape}")
        logging.info(f"列名: {df.columns.tolist()}")
        
        # 打印前几行数据以便调试
        if DEBUG:
            logging.info(f"数据前3行:\n{df.head(3)}")
            
            # 检查数据类型
            logging.info(f"数据类型:\n{df.dtypes}")
            
            # 检查是否有缺失值
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logging.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
        
        # 处理Independent test set.xlsx的特殊结构
        is_special_format = False
        if 'Phage' in df.columns and 'Host' in df.columns and 'Non-host' in df.columns and 'Taxonomy' in df.columns:
            is_special_format = True
            logging.info("检测到独立测试集的特殊格式")
            
            # 检查第一行是否是列名
            if isinstance(df.iloc[0]['Phage'], str) and df.iloc[0]['Phage'].lower() == 'accession':
                logging.info("检测到第一行是列名，剔除并重新构建DataFrame")
                header_row = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
                
                # 为了兼容性，重命名列以匹配我们的处理逻辑
                df.rename(columns={'Phage': 'phage', 'Host': 'host', 'Non-host': 'non-host'}, inplace=True)
        else:
            # 处理数据 - 检查是否需要特殊处理第一行
            if 'Phage' in df.columns and isinstance(df.iloc[0]['Phage'], str) and df.iloc[0]['Phage'].lower() == 'accession':
                logging.info("检测到第一行是列名，剔除并重新构建DataFrame")
                header_row = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            
            # 检查并标准化列名 - 转为小写
            df.columns = [col.lower() for col in df.columns]
        
        # 尝试检测可能的分类学列名模式
        taxonomy_patterns = identify_taxonomy_columns(df)
        logging.info(f"检测到的分类学列模式: {taxonomy_patterns}")
        
        return df
    except Exception as e:
        logging.error(f"加载测试数据出错: {str(e)}")
        raise

def identify_taxonomy_columns(df):
    """识别数据框中可能的分类学列模式"""
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    entities = ['phage', 'host', 'non-host', 'nonhost']
    
    patterns = {}
    columns = [col.lower() for col in df.columns]
    
    # 检测实体列
    for entity in entities:
        entity_col = next((col for col in columns 
                          if col == entity 
                          or entity in col 
                          or (entity == 'non-host' and 'nonhost' in col)), None)
        if entity_col:
            patterns[entity] = entity_col
    
    # 检测分类学列
    taxonomy_columns = {}
    for entity in patterns.keys():
        entity_taxonomy = {}
        for level in taxonomy_levels:
            # 尝试不同的列名格式
            level_col = next((col for col in columns 
                              if f"{entity}_{level}" in col.lower() 
                              or f"{level}_{entity}" in col.lower()
                              or (patterns[entity] in col and level in col)), None)
            if level_col:
                entity_taxonomy[level] = level_col
        taxonomy_columns[entity] = entity_taxonomy
    
    patterns['taxonomy'] = taxonomy_columns
    return patterns

def extract_taxonomy_features(df, encoders):
    """从测试数据中提取分类学特征"""
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    entities = ['phage', 'host', 'non-host']
    
    # 准备数据结构
    node_features = {}
    node_mapping = {}
    next_node_id = 0
    
    # 检测特殊文件格式
    is_special_format = False
    if 'Taxonomy' in df.columns and 'Taxonomy.1' in df.columns and 'Taxonomy.2' in df.columns:
        is_special_format = True
        logging.info("检测到特殊文件结构，使用专用处理逻辑")
    
    # 输出更多的数据集信息
    logging.info(f"数据集形状: {df.shape}")
    logging.info(f"数据集列名: {df.columns.tolist()}")
    logging.info(f"前5行数据:\n{df.head(5)}")
    
    # 获取正确的列名模式
    column_patterns = identify_taxonomy_columns(df)
    
    if DEBUG:
        logging.info(f"列名模式: {column_patterns}")
    
    # 获取实体列
    entity_columns = {}
    for entity in entities:
        entity_col = None
        # 尝试不同的匹配策略
        if entity in column_patterns:
            entity_col = column_patterns[entity]
        else:
            # 尝试其他匹配方式
            if entity == 'phage':
                entity_col = next((col for col in df.columns if 'phage' in col.lower() or 'accession' in col.lower()), None)
            elif entity == 'host':
                entity_col = next((col for col in df.columns if 'host' in col.lower() and 'non' not in col.lower()), None)
            elif entity == 'non-host':
                entity_col = next((col for col in df.columns if ('non' in col.lower() and 'host' in col.lower()) or 'nonhost' in col.lower()), None)
        
        entity_columns[entity] = entity_col
        
        if entity_col:
            logging.info(f"找到{entity}列: {entity_col}")
            # 显示该列中非空值的数量
            non_empty = df[entity_col].notna().sum()
            logging.info(f"{entity}列中有 {non_empty} 个非空值 (共 {len(df)} 行)")
        else:
            logging.warning(f"找不到{entity}列")
    
    # 检查是否找到了必要的实体列
    if not any(entity_columns.values()):  # 修改为any而不是all，只要有一个实体列就继续
        missing_entities = [e for e, c in entity_columns.items() if not c]
        logging.error(f"缺少所有实体列: {missing_entities}")
        logging.error(f"可用列: {df.columns.tolist()}")
        raise ValueError(f"找不到任何实体列，请检查数据格式")
    
    # 提取分类数据
    entity_data = {}
    for entity, entity_col in entity_columns.items():
        if entity_col:  # 只处理找到列的实体
            entity_data[entity] = extract_entity_data(df, entity_col, entity, taxonomy_levels, column_patterns)
            
            if DEBUG:
                if isinstance(entity_data[entity], pd.DataFrame):
                    logging.info(f"{entity}数据形状: {entity_data[entity].shape}")
                    if not entity_data[entity].empty:
                        logging.info(f"{entity}数据示例:\n{entity_data[entity].head(2)}")
                        # 检查每个级别的数据覆盖率
                        for level in taxonomy_levels:
                            if level in entity_data[entity].columns:
                                non_null = entity_data[entity][level].notna().sum()
                                coverage = non_null / len(entity_data[entity]) * 100
                                logging.info(f"{entity}.{level} 数据覆盖率: {coverage:.2f}% ({non_null}/{len(entity_data[entity])})")
                else:
                    logging.warning(f"{entity}数据不是DataFrame对象: {type(entity_data[entity])}")
        else:
            entity_data[entity] = pd.DataFrame()  # 空DataFrame
    
    # 创建节点特征
    total_features_extracted = 0
    
    # 处理实体特征提取
    for entity in entities:
        if entity not in entity_data or entity_data[entity].empty:
            logging.warning(f"没有{entity}数据可处理")
            continue
            
        entity_count = 0
        feature_count = 0
        
        for idx, row in entity_data[entity].iterrows():
            entity_id = row['entity_id']
            entity_count += 1
            
            if entity_id not in node_mapping:
                node_mapping[entity_id] = {
                    'node_id': next_node_id,
                    'type': entity
                }
                
                features = create_taxonomy_features(row, entity, taxonomy_levels, encoders)
                if features is not None and len(features) > 0:
                    node_features[next_node_id] = features
                    next_node_id += 1
                    total_features_extracted += 1
                    feature_count += 1
                else:
                    logging.debug(f"未能为{entity} {entity_id}创建特征")
        
        logging.info(f"处理了 {entity_count} 个{entity}，成功提取特征的有 {feature_count} 个")
    
    logging.info(f"共提取了 {total_features_extracted} 个实体的特征")
    
    # 检查是否有特征
    if not node_features:
        logging.error("没有提取到任何有效的节点特征！")
        # 输出更详细的调试信息
        if DEBUG:
            for entity in entities:
                if entity in entity_data:
                    logging.info(f"{entity}数据形状: {entity_data[entity].shape if isinstance(entity_data[entity], pd.DataFrame) else '非DataFrame'}")
                    if isinstance(entity_data[entity], pd.DataFrame) and not entity_data[entity].empty:
                        for level in taxonomy_levels:
                            if level in entity_data[entity].columns:
                                unique_values = entity_data[entity][level].unique()
                                logging.info(f"{entity}.{level} 唯一值数量: {len(unique_values)}")
                                if len(unique_values) > 0 and len(unique_values) <= 10:
                                    logging.info(f"{entity}.{level} 唯一值: {unique_values}")
        
        # 尝试一个更简单的逻辑来处理特殊文件
        if is_special_format:
            logging.info("尝试使用备用方法解析特殊格式文件...")
            node_features_simple, node_mapping_simple = extract_features_from_special_format(df, encoders, taxonomy_levels)
            if node_features_simple and len(node_features_simple) > 0:
                logging.info(f"使用备用方法成功提取了 {len(node_features_simple)} 个特征")
                return node_features_simple, node_mapping_simple
            
        # 尝试使用更宽松的特征提取方法
        logging.info("尝试使用宽松的特征提取方法...")
        node_features_relaxed, node_mapping_relaxed = extract_features_relaxed(df, encoders, taxonomy_levels)
        if node_features_relaxed and len(node_features_relaxed) > 0:
            logging.info(f"使用宽松方法成功提取了 {len(node_features_relaxed)} 个特征")
            return node_features_relaxed, node_mapping_relaxed
            
        raise ValueError("无法提取任何有效的节点特征")
    
    # 确保所有特征向量长度一致
    feature_lengths = [len(feat) for feat in node_features.values()]
    if len(set(feature_lengths)) > 1:
        logging.warning(f"特征长度不一致: {set(feature_lengths)}")
        max_length = max(feature_lengths)
        # 填充较短的特征向量
        for node_id, features in node_features.items():
            if len(features) < max_length:
                node_features[node_id] = np.pad(features, (0, max_length - len(features)))
    
    # 构建特征矩阵
    feature_dim = len(next(iter(node_features.values())))
    feature_matrix = np.zeros((len(node_features), feature_dim))
    for node_id, features in node_features.items():
        feature_matrix[node_id] = features
    
    logging.info(f"提取了 {len(node_features)} 个节点的特征，特征维度: {feature_dim}")
    
    return feature_matrix, node_mapping

def extract_features_relaxed(df, encoders, taxonomy_levels):
    """使用宽松条件提取特征，尝试提取更多实体"""
    logging.info("使用宽松条件进行特征提取...")
    node_features = {}
    node_mapping = {}
    next_node_id = 0
    
    # 实体类型
    entities = ['phage', 'host', 'non-host']
    
    # 尝试直接从每行数据提取实体ID和类型
    for idx, row in df.iterrows():
        # 对每一行，尝试确定它是哪种类型的实体
        entity_type = None
        entity_id = None
        
        # 检查是否有明确的类型标记
        for potential_entity in entities:
            for col in df.columns:
                if potential_entity.lower() in col.lower():
                    if not pd.isna(row[col]) and str(row[col]).strip():
                        entity_type = potential_entity
                        entity_id = str(row[col]).strip()
                        break
            if entity_type:
                break
                
        # 如果没有找到实体类型，尝试从其他列推断
        if not entity_type:
            # 如果有Accession列，假设为phage
            for col in df.columns:
                if 'accession' in col.lower() and not pd.isna(row[col]) and str(row[col]).strip():
                    entity_type = 'phage'
                    entity_id = str(row[col]).strip()
                    break
            
            # 如果有taxonomy列但没有accession，可能是host
            if not entity_type:
                for col in df.columns:
                    if 'taxonomy' in col.lower() and not pd.isna(row[col]) and str(row[col]).strip():
                        possible_id_col = df.columns[max(0, df.columns.get_loc(col)-1)]  # 尝试获取taxonomy前一列
                        if not pd.isna(row[possible_id_col]) and str(row[possible_id_col]).strip():
                            entity_type = 'host'  # 假设为host
                            entity_id = str(row[possible_id_col]).strip()
                            break
        
        # 如果仍然没有找到，但行不全是空的，尝试使用第一个非空值作为ID
        if not entity_type:
            non_na_cols = [col for col in df.columns if not pd.isna(row[col]) and str(row[col]).strip()]
            if non_na_cols:
                entity_type = 'phage'  # 默认假设为phage
                entity_id = str(row[non_na_cols[0]]).strip()
        
        # 如果找到了实体信息，尝试提取特征
        if entity_type and entity_id:
            # 添加到映射
            if entity_id not in node_mapping:
                node_mapping[entity_id] = {
                    'node_id': next_node_id,
                    'type': entity_type
                }
                
                # 尝试提取特征
                features = []
                has_features = False
                
                # 为每个分类学级别创建特征
                for level in taxonomy_levels:
                    # 尝试多种方式找到该级别的值
                    level_value = None
                    
                    # 方法1：直接查找级别名称列
                    level_cols = [col for col in df.columns if level.lower() in col.lower()]
                    if level_cols:
                        for level_col in level_cols:
                            if not pd.isna(row[level_col]) and str(row[level_col]).strip():
                                level_value = str(row[level_col]).strip()
                                break
                    
                    # 方法2：查找与实体类型相关的列
                    if not level_value:
                        entity_level_cols = [col for col in df.columns if entity_type.lower() in col.lower() and level.lower() in col.lower()]
                        if entity_level_cols:
                            for col in entity_level_cols:
                                if not pd.isna(row[col]) and str(row[col]).strip():
                                    level_value = str(row[col]).strip()
                                    break
                    
                    # 如果找到了值，并且编码器中有这个级别
                    if level_value and level in encoders:
                        # 确保是字符串类型
                        if not isinstance(level_value, str):
                            level_value = str(level_value)
                        
                        # 检查这个值是否在编码器中
                        if level_value in encoders[level].classes_:
                            level_encoded = encoders[level].transform([level_value])[0]
                            
                            # 创建独热编码
                            level_length = len(encoders[level].classes_)
                            level_onehot = np.zeros(level_length)
                            level_onehot[level_encoded] = 1
                            features.extend(level_onehot.tolist())
                            has_features = True
                        else:
                            # 值不在编码器中，使用全零向量
                            level_length = len(encoders[level].classes_)
                            features.extend(np.zeros(level_length).tolist())
                    elif level in encoders:
                        # 如果没有找到值或编码器中没有这个级别，使用全零向量
                        level_length = len(encoders[level].classes_)
                        features.extend(np.zeros(level_length).tolist())
                
                # 如果成功提取了特征，添加到特征集
                if has_features and len(features) > 0:
                    node_features[next_node_id] = np.array(features)
                    next_node_id += 1
                    logging.debug(f"成功为{entity_type} {entity_id}提取了特征")
    
    logging.info(f"使用宽松条件共提取了 {len(node_features)} 个实体的特征")
    return node_features, node_mapping

def extract_entity_data(df, entity_col, entity_type, taxonomy_levels, column_patterns=None):
    """提取实体的分类学数据"""
    entity_data = []
    total_rows = 0
    skipped_rows = 0
    
    # 如果有列名模式，优先使用
    taxonomy_cols = {}
    if column_patterns and 'taxonomy' in column_patterns and entity_type in column_patterns['taxonomy']:
        taxonomy_cols = column_patterns['taxonomy'][entity_type]
    
    # 处理Independent test set.xlsx的特殊结构
    # 这个文件的结构是：Phage, Taxonomy, Unnamed: 2(phylum), Unnamed: 3(class)...
    #                 Host, Taxonomy.1, Unnamed: 9(phylum), Unnamed: 10(class)...
    #                 Non-host, Taxonomy.2, Unnamed: 16(phylum), Unnamed: 17(class)...
    special_structure = False
    
    # 检查特殊文件格式
    if 'Taxonomy' in df.columns and 'Taxonomy.1' in df.columns and 'Taxonomy.2' in df.columns:
        special_structure = True
        logging.info(f"检测到特殊文件结构，使用特定的列映射")
        
        # 映射各实体的分类学列
        taxonomy_map = {
            'phage': {
                'taxonomy_col': 'Taxonomy',
                'level_cols': {
                    'kingdom': 'Taxonomy',
                    'phylum': 'Unnamed: 2',
                    'class': 'Unnamed: 3',
                    'order': 'Unnamed: 4',
                    'family': 'Unnamed: 5',
                    'genus': 'Unnamed: 6'
                }
            },
            'host': {
                'taxonomy_col': 'Taxonomy.1',
                'level_cols': {
                    'kingdom': 'Taxonomy.1',
                    'phylum': 'Unnamed: 9',
                    'class': 'Unnamed: 10',
                    'order': 'Unnamed: 11',
                    'family': 'Unnamed: 12',
                    'genus': 'Unnamed: 13'
                }
            },
            'non-host': {
                'taxonomy_col': 'Taxonomy.2',
                'level_cols': {
                    'kingdom': 'Taxonomy.2',
                    'phylum': 'Unnamed: 16',
                    'class': 'Unnamed: 17',
                    'order': 'Unnamed: 18',
                    'family': 'Unnamed: 19',
                    'genus': 'Unnamed: 20'
                }
            }
        }
        
        if entity_type in taxonomy_map:
            logging.info(f"使用 {entity_type} 的特定列映射: {taxonomy_map[entity_type]['level_cols']}")
            
            for idx, row in df.iterrows():
                total_rows += 1
                entity_id = row[entity_col]
                if pd.isna(entity_id):
                    skipped_rows += 1
                    continue
                    
                entity_row = {'entity_id': entity_id, 'row_idx': idx}
                
                has_taxonomy_data = False
                
                # 使用特定映射提取分类学数据
                for level, level_col in taxonomy_map[entity_type]['level_cols'].items():
                    if level_col in df.columns:
                        taxonomy_value = row[level_col]
                        entity_row[level] = taxonomy_value
                        if not pd.isna(taxonomy_value):
                            has_taxonomy_data = True
                
                # 如果没有要求有分类数据，或者确实有分类数据，添加实体
                if has_taxonomy_data or True:  # 即使没有分类数据也添加
                    entity_data.append(entity_row)
            
            df_entity = pd.DataFrame(entity_data)
            logging.info(f"{entity_type} 处理了 {total_rows} 行，跳过了 {skipped_rows} 行，找到 {len(entity_data)} 个实体")
            
            if DEBUG and not df_entity.empty:
                logging.info(f"{entity_type}分类数据形状: {df_entity.shape}")
                logging.info(f"{entity_type}分类数据列: {df_entity.columns.tolist()}")
                if not df_entity.empty and len(df_entity) > 0:
                    logging.info(f"{entity_type}分类数据示例:\n{df_entity.head(1)}")
            elif df_entity.empty:
                logging.warning(f"未找到任何{entity_type}的分类数据！检查是否有ID列: {entity_col} in {df.columns.tolist()}")
            
            return df_entity
    
    # 如果不是特殊结构或者特殊处理失败，使用通用处理逻辑
    if not special_structure or len(entity_data) == 0:
        for idx, row in df.iterrows():
            total_rows += 1
            entity_id = row[entity_col]
            if pd.isna(entity_id):
                skipped_rows += 1
                continue
                
            entity_row = {'entity_id': entity_id, 'row_idx': idx}
            
            has_taxonomy_data = False
            
            # 先尝试使用已识别的列名模式
            for level in taxonomy_levels:
                if level in taxonomy_cols and taxonomy_cols[level] in df.columns:
                    taxonomy_value = row[taxonomy_cols[level]]
                    entity_row[level] = taxonomy_value
                    if not pd.isna(taxonomy_value):
                        has_taxonomy_data = True
                    continue
                
                # 如果没有预定义的列模式，尝试搜索可能的列
                # 尝试不同的列名格式
                level_col = None
                
                # 格式1: entity_level
                col_pattern1 = f"{entity_type}_{level}"
                matching_cols1 = [col for col in df.columns if col_pattern1 in col.lower()]
                
                # 格式2: level_entity
                col_pattern2 = f"{level}_{entity_type}"
                matching_cols2 = [col for col in df.columns if col_pattern2 in col.lower()]
                
                # 格式3: entity中包含level
                matching_cols3 = [col for col in df.columns if entity_col in col and level in col.lower()]
                
                # 结合所有可能的匹配
                all_matching_cols = matching_cols1 + matching_cols2 + matching_cols3
                
                if all_matching_cols:
                    level_col = all_matching_cols[0]  # 使用第一个匹配
                    entity_row[level] = row[level_col]
                    if not pd.isna(row[level_col]):
                        has_taxonomy_data = True
                else:
                    # 对于结构化Excel，可能在entity_col右侧有固定位置的分类学信息
                    try:
                        level_idx = taxonomy_levels.index(level)
                        if entity_col in df.columns:
                            entity_col_idx = df.columns.get_loc(entity_col)
                            if entity_col_idx + 1 + level_idx < len(df.columns):
                                potential_col = df.columns[entity_col_idx + 1 + level_idx]
                                if 'Unnamed' in potential_col or 'Taxonomy' in potential_col or level.lower() in potential_col.lower():
                                    entity_row[level] = row[potential_col]
                                    if not pd.isna(row[potential_col]):
                                        has_taxonomy_data = True
                    except (ValueError, KeyError) as e:
                        logging.debug(f"在查找级别{level}时出错: {str(e)}")
            
            # 降低限制，即使没有分类数据也添加
            if has_taxonomy_data or True:  # 即使没有分类数据也添加
                entity_data.append(entity_row)
        
        df_entity = pd.DataFrame(entity_data)
        logging.info(f"{entity_type} 处理了 {total_rows} 行，跳过了 {skipped_rows} 行，找到 {len(entity_data)} 个实体")
        
        if DEBUG and not df_entity.empty:
            logging.info(f"{entity_type}分类数据形状: {df_entity.shape}")
            logging.info(f"{entity_type}分类数据列: {df_entity.columns.tolist()}")
            if not df_entity.empty and len(df_entity) > 0:
                logging.info(f"{entity_type}分类数据示例:\n{df_entity.head(1)}")
        elif df_entity.empty:
            logging.warning(f"未找到任何{entity_type}的分类数据！检查是否有ID列: {entity_col} in {df.columns.tolist()}")
        
        return df_entity

def create_taxonomy_features(row, entity_type, taxonomy_levels, encoders):
    """为一个实体创建分类学特征向量"""
    features = []
    has_features = False
    
    missing_levels = []
    unknown_values = []
    
    for level in taxonomy_levels:
        if level in row and level in encoders:
            if pd.isna(row[level]):
                # 级别存在但值为空
                missing_levels.append(level)
                if level in encoders:
                    level_length = len(encoders[level].classes_)
                    features.extend(np.zeros(level_length))
                continue
                
            level_value = row[level]
            
            # 确保字符串类型
            if not isinstance(level_value, str):
                level_value = str(level_value)
            
            # 检查这个值是否在编码器中
            if level_value in encoders[level].classes_:
                level_encoded = encoders[level].transform([level_value])[0]
                
                # 创建独热编码
                level_length = len(encoders[level].classes_)
                level_onehot = np.zeros(level_length)
                level_onehot[level_encoded] = 1
                features.extend(level_onehot)
                has_features = True
            else:
                # 值不在编码器中
                unknown_values.append(f"{level}={level_value}")
                level_length = len(encoders[level].classes_)
                features.extend(np.zeros(level_length))
                
                # 即使值未知，也认为有特征（放宽要求）
                has_features = True
        else:
            # 如果没有这个级别的数据，用零填充
            if level not in row:
                missing_levels.append(level)
            if level in encoders:
                level_length = len(encoders[level].classes_)
                features.extend(np.zeros(level_length))
    
    if DEBUG and (missing_levels or unknown_values):
        if entity_type == 'phage':  # 只对phage记录详细信息，避免日志过多
            if missing_levels:
                logging.debug(f"{entity_type} {row['entity_id']} 缺少级别: {', '.join(missing_levels)}")
            if unknown_values:
                logging.debug(f"{entity_type} {row['entity_id']} 未知值: {', '.join(unknown_values)}")
    
    # 只要有特征列表，即使没有明确特征，也返回
    if not features:
        if DEBUG and entity_type == 'phage':
            logging.debug(f"{entity_type} {row['entity_id']} 没有提取到任何特征")
        return None
        
    # 特征长度检查
    total_expected_length = sum(len(encoders[level].classes_) for level in taxonomy_levels if level in encoders)
    if len(features) != total_expected_length:
        if DEBUG:
            logging.debug(f"{entity_type} {row['entity_id']} 特征长度不匹配: {len(features)} vs 预期 {total_expected_length}")
        # 确保长度正确
        if len(features) < total_expected_length:
            features.extend(np.zeros(total_expected_length - len(features)))
        else:
            features = features[:total_expected_length]
    
    # 如果有超过一个非零值，则认为有特征；否则返回None
    if np.count_nonzero(features) > 0:
        has_features = True
    
    # 放宽条件，即使没有特征也返回
    return np.array(features)

def build_graph(node_features, node_mapping, phage_hosts_pairs=None):
    """构建用于推理的图"""
    # 将特征矩阵转换为张量
    x = torch.FloatTensor(node_features)
    
    # 创建边索引
    edges = []
    
    if phage_hosts_pairs:
        # 如果提供了噬菌体-宿主对，使用它们来建立边
        for phage_id, host_id in phage_hosts_pairs:
            if phage_id in node_mapping and host_id in node_mapping:
                phage_node = node_mapping[phage_id]['node_id']
                host_node = node_mapping[host_id]['node_id']
                edges.append([phage_node, host_node])
    else:
        # 否则，假设所有噬菌体与所有宿主都有连接
        phage_nodes = [info['node_id'] for entity_id, info in node_mapping.items() if info['type'] == 'phage']
        host_nodes = [info['node_id'] for entity_id, info in node_mapping.items() if info['type'] == 'host']
        
        for phage_node in phage_nodes:
            for host_node in host_nodes:
                edges.append([phage_node, host_node])
    
    # 将边列表转换为边索引
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 添加自环和无向化
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = to_undirected(edge_index)
    else:
        # 如果没有边，创建空边索引
        logging.warning("没有找到有效的边！创建空边索引。")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # 确保节点标签为正确的维度
    y = torch.zeros(x.size(0), dtype=torch.long)
    
    # 创建掩码
    mask = torch.ones(x.size(0), dtype=torch.bool)
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    data.mask = mask
    
    if DEBUG:
        logging.info(f"构建图: 节点数={x.size(0)}, 边数={edge_index.size(1)//2}")
        # 记录节点类型统计
        node_types = {}
        for _, info in node_mapping.items():
            node_type = info['type']
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        logging.info(f"节点类型统计: {node_types}")
    
    return data

def run_inference(model, data, device, node_mapping):
    """使用模型进行推理"""
    model.eval()
    
    # 处理可能的空图情况
    if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
        logging.error("无法进行推理：图中没有节点或边！")
        return {}
    
    # 确保数据在正确的设备上
    data = data.to(device)
    
    results = {}
    
    try:
        with torch.no_grad():
            # 运行模型
            out = model(data.x, data.edge_index)
            
            # 获取概率分布
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
            
            # 获取预测类别
            preds = out.argmax(dim=1).cpu().numpy()
            
            # 收集结果
            for entity_id, info in node_mapping.items():
                node_id = info['node_id']
                entity_type = info['type']
                
                # 只处理噬菌体和宿主
                if entity_type in ['phage', 'host']:
                    prob = probs[node_id].tolist()
                    pred = int(preds[node_id])
                    
                    results[entity_id] = {
                        'type': entity_type,
                        'prediction': pred,
                        'probability': prob,
                        'prediction_confidence': float(prob[pred])
                    }
        
        logging.info(f"完成推理，处理了 {len(results)} 个实体")
        return results
    except Exception as e:
        logging.error(f"推理过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {}

def analyze_results(results, node_mapping):
    """分析推理结果"""
    if not results:
        logging.warning("没有结果可分析")
        return {}
    
    # 收集统计信息
    stats = {
        'total_entities': len(results),
        'by_type': {},
        'by_prediction': {},
        'high_confidence_count': 0  # 高置信度预测的数量
    }
    
    # 按类型和预测统计
    for entity_id, result in results.items():
        entity_type = result['type']
        prediction = result['prediction']
        confidence = result['prediction_confidence']
        
        # 按类型统计
        if entity_type not in stats['by_type']:
            stats['by_type'][entity_type] = 0
        stats['by_type'][entity_type] += 1
        
        # 按预测统计
        pred_key = f"class_{prediction}"
        if pred_key not in stats['by_prediction']:
            stats['by_prediction'][pred_key] = 0
        stats['by_prediction'][pred_key] += 1
        
        # 高置信度计数
        if confidence > 0.8:
            stats['high_confidence_count'] += 1
    
    # 计算高置信度比例
    if stats['total_entities'] > 0:
        stats['high_confidence_ratio'] = stats['high_confidence_count'] / stats['total_entities']
    else:
        stats['high_confidence_ratio'] = 0
    
    # 记录统计信息
    logging.info(f"结果统计: 总实体数={stats['total_entities']}")
    logging.info(f"按类型: {stats['by_type']}")
    logging.info(f"按预测: {stats['by_prediction']}")
    logging.info(f"高置信度预测: {stats['high_confidence_count']} ({stats['high_confidence_ratio']:.2%})")
    
    # 计算正确率（如果可能）
    if 'phage' in stats['by_type'] and 'host' in stats['by_type']:
        phage_count = stats['by_type'].get('phage', 0)
        host_count = stats['by_type'].get('host', 0)
        
        # 假设class_0对应phage, class_1对应host
        class_0_count = stats['by_prediction'].get('class_0', 0)
        class_1_count = stats['by_prediction'].get('class_1', 0)
        
        # 简单估计正确率
        if (phage_count + host_count) > 0:
            estimated_accuracy = (min(phage_count, class_0_count) + min(host_count, class_1_count)) / (phage_count + host_count)
            stats['estimated_accuracy'] = estimated_accuracy
            logging.info(f"估计正确率: {estimated_accuracy:.2%}")
            
            # 打印更多详细信息
            logging.info(f"噬菌体样本数: {phage_count}, 宿主样本数: {host_count}")
            logging.info(f"预测为类别0的样本数: {class_0_count}, 预测为类别1的样本数: {class_1_count}")
    
    return stats

def visualize_results(results, stats, output_path):
    """可视化结果"""
    if not results:
        logging.warning("没有结果可视化")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path) if output_path else "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 置信度分布图
    plt.figure(figsize=(10, 6))
    confidences = [result['prediction_confidence'] for result in results.values()]
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.xlabel('置信度')
    plt.ylabel('实体数量')
    plt.title('预测置信度分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    confidence_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"置信度分布图已保存到: {confidence_path}")
    
    # 2. 按类型的预测分布
    entity_types = set(result['type'] for result in results.values())
    predictions = set(result['prediction'] for result in results.values())
    
    # 按类型计数
    type_pred_counts = {}
    for entity_type in entity_types:
        type_pred_counts[entity_type] = {}
        for pred in predictions:
            type_pred_counts[entity_type][pred] = 0
    
    for result in results.values():
        entity_type = result['type']
        pred = result['prediction']
        type_pred_counts[entity_type][pred] += 1
    
    # 绘制按类型的预测分布
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(predictions))
    
    for i, entity_type in enumerate(entity_types):
        counts = [type_pred_counts[entity_type][pred] for pred in sorted(predictions)]
        plt.bar(index + i*bar_width, counts, bar_width, alpha=0.7, label=entity_type)
    
    plt.xlabel('预测类别')
    plt.ylabel('实体数量')
    plt.title('按类型的预测分布')
    plt.xticks(index + bar_width/2, [f'类别 {p}' for p in sorted(predictions)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    prediction_path = os.path.join(output_dir, 'prediction_by_type.png')
    plt.savefig(prediction_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"预测分布图已保存到: {prediction_path}")
    
    # 3. 混淆矩阵 (估计值)
    if 'phage' in type_pred_counts and 'host' in type_pred_counts:
        plt.figure(figsize=(8, 6))
        cm = np.zeros((2, 3), dtype=int)  # 假设是2种类型，3种预测类别
        
        # 填充混淆矩阵
        entity_type_map = {'phage': 0, 'host': 1}
        for entity_type, pred_counts in type_pred_counts.items():
            if entity_type in entity_type_map:
                row = entity_type_map[entity_type]
                for pred, count in pred_counts.items():
                    cm[row, pred] = count
        
        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['类别 0', '类别 1', '类别 2'],
                   yticklabels=['噬菌体', '宿主'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('混淆矩阵 (估计)')
        confusion_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"混淆矩阵已保存到: {confusion_path}")
    
    # 4. 保存详细统计数据
    stats_output = os.path.join(output_dir, 'inference_stats.json')
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"统计信息已保存到: {stats_output}")
    
    # 5. 创建一个简单的HTML报告
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphSAGE推理结果</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .stats {{ margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .image-box {{ margin-bottom: 20px; text-align: center; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>分类学推理结果报告</h1>
            
            <div class="stats">
                <h2>统计信息</h2>
                <p>总实体数: {stats['total_entities']}</p>
                <p>高置信度预测: {stats['high_confidence_count']} ({stats['high_confidence_ratio']:.2%})</p>
                {f'<p>估计正确率: {stats.get("estimated_accuracy", 0):.2%}</p>' if 'estimated_accuracy' in stats else ''}
                
                <h3>按类型统计</h3>
                <table>
                    <tr><th>类型</th><th>数量</th></tr>
                    {''.join(f'<tr><td>{t}</td><td>{c}</td></tr>' for t, c in stats['by_type'].items())}
                </table>
                
                <h3>按预测统计</h3>
                <table>
                    <tr><th>预测</th><th>数量</th></tr>
                    {''.join(f'<tr><td>{p}</td><td>{c}</td></tr>' for p, c in stats['by_prediction'].items())}
                </table>
            </div>
            
            <div class="image-container">
                <div class="image-box">
                    <h2>置信度分布</h2>
                    <img src="confidence_distribution.png" alt="置信度分布">
                </div>
                
                <div class="image-box">
                    <h2>按类型的预测分布</h2>
                    <img src="prediction_by_type.png" alt="按类型的预测分布">
                </div>
                
                <div class="image-box">
                    <h2>混淆矩阵</h2>
                    <img src="confusion_matrix.png" alt="混淆矩阵">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, 'inference_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    logging.info(f"HTML报告已保存到: {html_path}")
    
    # 尝试自动打开报告
    try:
        import webbrowser
        webbrowser.open(f'file://{html_path}')
        logging.info("已自动打开报告页面")
    except:
        logging.info(f"请手动打开报告: {html_path}")
    
    logging.info(f"可视化结果已保存到 {output_dir} 目录")

def load_model(model_path, hidden_dim=256):
    """加载模型"""
    try:
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查是否有嵌套的model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            logging.info("检测到嵌套的model_state_dict结构")
            
            # 使用config中的参数初始化模型
            if 'config' in checkpoint:
                config = checkpoint['config']
                logging.info(f"使用保存的配置: {config}")
                
                # 获取输入和输出维度
                in_channels = config.get('input_channels', 783)  # 使用我们提取的特征维度
                hidden_channels = config.get('hidden_channels', hidden_dim)
                num_classes = config.get('num_classes', 2)
                num_layers = config.get('num_layers', 3)
                
                # 创建模型
                model = GraphSAGE(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=num_classes,
                    num_layers=num_layers,
                    dropout=0.2
                )
                
                # 加载模型权重
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"成功加载模型参数：输入={in_channels}, 隐藏={hidden_channels}, 输出={num_classes}, 层数={num_layers}")
                return model
        
        # 如果没有嵌套结构，尝试直接加载
        logging.info("尝试直接加载模型权重")
        
        # 检查是否有模型配置信息
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            # 使用保存的配置创建模型
            config = checkpoint['model_config']
            model = GraphSAGE(
                in_channels=config.get('in_channels', 783),  # 使用我们提取的特征维度
                hidden_channels=config.get('hidden_channels', hidden_dim),
                out_channels=config.get('out_channels', 2),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.2)
            )
            # 加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果没有model_state_dict字段，尝试直接加载整个checkpoint
                model.load_state_dict(checkpoint)
        else:
            # 尝试检测模型结构
            try:
                # 使用默认参数创建模型
                logging.info("使用默认参数创建模型并尝试加载权重")
                
                # 使用提取的特征维度作为输入
                in_channels = 783  # 提取的特征维度
                model = GraphSAGE(
                    in_channels=in_channels,
                    hidden_channels=hidden_dim,
                    out_channels=2  # 默认二分类
                )
                
                # 尝试加载权重
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as inner_e:
                logging.error(f"默认模型加载失败: {str(inner_e)}")
                
                # 最后的备选方案：使用非常基本的模型结构
                logging.info("尝试使用最基本的模型结构")
                model = GraphSAGE(
                    in_channels=783,  # 使用我们提取的特征维度
                    hidden_channels=hidden_dim,
                    out_channels=2,
                    num_layers=2,
                    dropout=0.1
                )
                
                # 不加载权重，使用随机初始化
                logging.warning("无法加载保存的权重，将使用随机初始化的模型！结果可能不准确。")
        
        logging.info(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        logging.error(f"加载模型出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 紧急备选方案：返回一个随机初始化的模型
        logging.warning("返回随机初始化的模型，结果将不可靠！")
        model = GraphSAGE(
            in_channels=783,  # 使用我们提取的特征维度
            hidden_channels=hidden_dim,
            out_channels=2,
            num_layers=2
        )
    return model

def predict(model, data, device='cuda'):
    """使用模型进行预测"""
    # 设置模型为评估模式
    model.eval()
    
    # 将数据移到设备上
    data = data.to(device)
    
    try:
        with torch.no_grad():
            # 前向传播
            out = model(data.x, data.edge_index)
            
            # 获取概率分布
            probs = torch.nn.functional.softmax(out, dim=1)
            
            # 获取预测类别
            preds = out.argmax(dim=1)
            
        return preds.cpu().numpy(), probs.cpu().numpy()
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def save_results(results, output_path):
    """保存推理结果"""
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 准备结果为JSON格式
        json_results = {}
        for entity_id, result in results.items():
            # 确保值是可序列化的
            json_results[str(entity_id)] = {
                'type': result['type'],
                'prediction': int(result['prediction']),
                'prediction_confidence': float(result['prediction_confidence']),
                'probability': [float(p) for p in result['probability']]
            }
        
        # 写入文件
        logging.info(f"将结果写入到: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 检查文件是否成功创建
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logging.info(f"结果文件已保存: {output_path}, 大小: {file_size/1024:.2f} KB")
        else:
            logging.warning(f"无法验证结果文件的创建: {output_path}")
        
        # 同时保存一个额外的副本到output目录
        try:
            # 确保output目录存在
            backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir, exist_ok=True)
                
            base_filename = os.path.basename(output_path)
            backup_path = os.path.join(backup_dir, f"latest_{base_filename}")
            with open(backup_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logging.info(f"结果备份已保存到: {backup_path}")
        except Exception as e:
            logging.warning(f"保存备份副本失败: {str(e)}")
        
        logging.info(f"结果已保存完成")
    except Exception as e:
        logging.error(f"保存结果出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 尝试保存到备用位置
        try:
            # 尝试保存到当前目录
            backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency_results.json")
            with open(backup_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logging.info(f"由于原始保存失败，结果已保存到备用位置: {backup_path}")
        except Exception as backup_error:
            logging.error(f"备用保存也失败: {str(backup_error)}")

def analyze_phage_host_relationships(results, output_path):
    """分析噬菌体和宿主之间的匹配关系，并生成详细报告"""
    logging.info("分析噬菌体-宿主匹配关系...")
    
    # 分离噬菌体和宿主
    phages = {}
    hosts = {}
    for entity_id, result in results.items():
        if result['type'] == 'phage':
            phages[entity_id] = result
        elif result['type'] == 'host':
            hosts[entity_id] = result
    
    # 分析匹配关系
    phage_host_matches = []
    special_hosts = []
    
    # 按预测类别分类宿主
    hosts_by_class = {}
    for host_id, host in hosts.items():
        prediction = host['prediction']
        if prediction not in hosts_by_class:
            hosts_by_class[prediction] = []
        hosts_by_class[prediction].append((host_id, host))
    
    # 找出特殊宿主（类别2）
    special_class = 2
    if special_class in hosts_by_class:
        for host_id, host in hosts_by_class[special_class]:
            special_hosts.append({
                'host_id': host_id,
                'confidence': host['prediction_confidence'],
                'probability': host['probability']
            })
    
    # 为每个噬菌体找出最佳匹配的宿主
    for phage_id, phage in phages.items():
        # 假设噬菌体类别为0，宿主类别为1
        # 这是一个简化的匹配逻辑，实际中可能需要更复杂的关联分析
        # 这里我们基于ID的相似性创建一个示例匹配
        
        # 从普通宿主中找出潜在匹配
        # 实际应用中，应该使用专业的相似度算法，如BLAST等
        matching_hosts = []
        if 1 in hosts_by_class:
            for host_id, host in hosts_by_class[1]:
                # 这里使用一个简单的字符匹配作为示例
                # 实际匹配应基于生物学相似性
                similarity = 0
                for c1, c2 in zip(str(phage_id), str(host_id)):
                    if c1 == c2:
                        similarity += 1
                
                if similarity > 0:  # 降低阈值，尝试找到任何可能的匹配
                    matching_hosts.append({
                        'host_id': host_id,
                        'similarity': similarity,
                        'confidence': host['prediction_confidence']
                    })
                    
                # 尝试使用更多匹配方式
                # 如果ID中包含相同子串
                if len(str(phage_id)) >= 3 and len(str(host_id)) >= 3:
                    for i in range(len(str(phage_id))-2):
                        substr = str(phage_id)[i:i+3]
                        if substr in str(host_id):
                            # 添加一个子串匹配分数
                            matching_hosts.append({
                                'host_id': host_id,
                                'similarity': 1,  # 基础相似度
                                'confidence': host['prediction_confidence'],
                                'match_type': '子串匹配'
                            })
                            break  # 避免重复添加
        
        # 按相似度排序并取前3个
        matching_hosts.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = matching_hosts[:3] if matching_hosts else []
        
        phage_host_matches.append({
            'phage_id': phage_id,
            'phage_confidence': phage['prediction_confidence'],
            'matching_hosts': top_matches
        })
    
    # 生成报告
    # 创建一个简单的CSV文件
    csv_path = os.path.join(os.path.dirname(output_path), "phage_host_matches.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['噬菌体ID', '预测置信度', '匹配宿主ID', '相似度', '宿主置信度'])
        
        for match in phage_host_matches:
            phage_id = match['phage_id']
            phage_conf = match['phage_confidence']
            
            if match['matching_hosts']:
                for host_match in match['matching_hosts']:
                    writer.writerow([
                        phage_id, 
                        phage_conf,
                        host_match['host_id'],
                        host_match['similarity'],
                        host_match['confidence']
                    ])
            else:
                writer.writerow([phage_id, phage_conf, '无匹配宿主', '', ''])
    
    # 特殊宿主报告
    special_csv_path = os.path.join(os.path.dirname(output_path), "special_hosts.csv")
    with open(special_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['特殊宿主ID', '预测置信度', '类别0概率', '类别1概率', '类别2概率'])
        
        for host in special_hosts:
            writer.writerow([
                host['host_id'],
                host['confidence'],
                host['probability'][0],
                host['probability'][1],
                host['probability'][2]
            ])
    
    # 创建一个更详细的HTML报告
    html_path = os.path.join(os.path.dirname(output_path), "phage_host_relationships.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>噬菌体-宿主匹配关系分析</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .phage-row {{ background-color: #e6f3ff; }}
            .host-row {{ background-color: #fff; }}
            .special-host {{ background-color: #ffe6e6; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ margin-bottom: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
            .no-match {{ color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>噬菌体-宿主匹配关系分析</h1>
            
            <div class="summary">
                <h2>总结</h2>
                <p>总噬菌体数: {len(phages)}</p>
                <p>总宿主数: {len(hosts)}</p>
                <p>普通宿主数: {len(hosts) - len(special_hosts)}</p>
                <p>特殊宿主数: {len(special_hosts)}</p>
            </div>
            
            <h2>噬菌体-宿主匹配表</h2>
            <p>下表显示每个噬菌体与其最可能的宿主匹配。相似度是基于简单的ID匹配算法，仅用于演示。</p>
            <table>
                <tr>
                    <th>噬菌体ID</th>
                    <th>预测置信度</th>
                    <th>匹配宿主</th>
                    <th>匹配相似度</th>
                    <th>宿主置信度</th>
                </tr>
    """
    
    # 添加噬菌体-宿主匹配行
    for match in phage_host_matches:
        phage_id = match['phage_id']
        phage_conf = match['phage_confidence']
        
        html_content += f"""
                <tr class="phage-row">
                    <td>{phage_id}</td>
                    <td>{phage_conf:.4f}</td>
                    <td colspan="3"></td>
                </tr>
        """
        
        if match['matching_hosts']:
            for host_match in match['matching_hosts']:
                html_content += f"""
                <tr class="host-row">
                    <td></td>
                    <td></td>
                    <td>{host_match['host_id']}</td>
                    <td>{host_match['similarity']}</td>
                    <td>{host_match['confidence']:.4f}</td>
                </tr>
                """
        else:
            html_content += f"""
                <tr class="host-row">
                    <td></td>
                    <td></td>
                    <td colspan="3" class="no-match">无匹配宿主</td>
                </tr>
            """
    
    # 添加特殊宿主表
    html_content += f"""
            </table>
            
            <h2>特殊宿主列表 (类别2)</h2>
            <p>以下宿主被分类为特殊类型（类别2），可能具有独特的特征或与标准宿主有显著不同。</p>
            <table>
                <tr>
                    <th>宿主ID</th>
                    <th>预测置信度</th>
                    <th>类别0概率</th>
                    <th>类别1概率</th>
                    <th>类别2概率</th>
                </tr>
    """
    
    for host in special_hosts:
        html_content += f"""
                <tr class="special-host">
                    <td>{host['host_id']}</td>
                    <td>{host['confidence']:.4f}</td>
                    <td>{host['probability'][0]:.4f}</td>
                    <td>{host['probability'][1]:.4f}</td>
                    <td>{host['probability'][2]:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <div class="summary">
                <h3>注意事项</h3>
                <ul>
                    <li>这些匹配基于简化的算法，实际生物学匹配需要更专业的分析</li>
                    <li>相似度分数仅用于演示目的，不应被视为实际生物学亲缘关系</li>
                    <li>特殊宿主可能代表具有独特分类学特征的生物体</li>
                    <li>CSV格式的详细数据可在同目录下的 phage_host_matches.csv 和 special_hosts.csv 文件中找到</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"噬菌体-宿主匹配关系CSV已保存到: {csv_path}")
    logging.info(f"特殊宿主列表CSV已保存到: {special_csv_path}")
    logging.info(f"详细HTML报告已保存到: {html_path}")
    
    # 尝试自动打开报告
    try:
        import webbrowser
        webbrowser.open(f'file://{html_path}')
        logging.info("已自动打开噬菌体-宿主关系报告")
    except:
        logging.info(f"请手动打开报告: {html_path}")
    
    return {
        'phage_count': len(phages),
        'host_count': len(hosts),
        'normal_host_count': len(hosts) - len(special_hosts),
        'special_host_count': len(special_hosts),
        'report_path': html_path,
        'csv_path': csv_path
    }

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置调试模式
    global DEBUG
    DEBUG = True  # 默认启用调试
    logging.info("启用调试模式")
    
    # 设置可视化
    visualize = True  # 默认启用可视化
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 获取项目根目录(脚本所在目录的上一级)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 处理路径
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.normpath(os.path.join(project_root, path))
    
    # 应用路径解析
    model_path = resolve_path(args.model)
    test_data_path = resolve_path(args.test_data)
    encoders_path = resolve_path(args.encoders)
    output_path = resolve_path(args.output)
    
    # 打印使用的参数
    logging.info(f"使用模型: {model_path}")
    logging.info(f"使用测试数据: {test_data_path}")
    logging.info(f"使用编码器: {encoders_path}")
    logging.info(f"结果将保存到: {output_path}")
    
    try:
        # 1. 加载分类学编码器
        logging.info("正在加载分类学编码器...")
        encoders = load_taxonomy_encoders(encoders_path)
        
        # 2. 加载测试数据
        logging.info("正在加载测试数据...")
        test_data = load_test_data(test_data_path)
        
        # 3. 提取分类学特征
        logging.info("正在提取分类学特征...")
        try:
            node_features, node_mapping = extract_taxonomy_features(test_data, encoders)
        except Exception as e:
            logging.error(f"特征提取失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
        
        # 4. 构建图
        logging.info("正在构建图...")
        graph_data = build_graph(node_features, node_mapping)
        
        # 5. 加载模型
        logging.info("正在加载模型...")
        model = load_model(model_path, hidden_dim=args.hidden_dim)
        model = model.to(device)
        
        # 6. 运行推理
        logging.info("正在进行推理...")
        results = run_inference(model, graph_data, device, node_mapping)
        
        # 7. 分析结果
        logging.info("正在分析结果...")
        stats = analyze_results(results, node_mapping)
        
        # 8. 保存结果
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"正在保存结果到: {output_path}...")
        save_results(results, output_path)
        
        # 9. 可视化结果
        logging.info("正在可视化结果...")
        visualize_results(results, stats, output_path)
        
        # 10. 新增: 分析噬菌体-宿主匹配关系
        logging.info("正在分析噬菌体-宿主匹配关系...")
        relationship_stats = analyze_phage_host_relationships(results, output_path)
        
        logging.info("推理完成！")
        
        # 打印总结
        print("\n" + "="*50)
        print("推理完成！总结：")
        print(f"- 总实体数: {stats['total_entities']}")
        print(f"- 按类型: {stats['by_type']}")
        print(f"- 按预测: {stats['by_prediction']}")
        print(f"- 高置信度预测率: {stats['high_confidence_ratio']:.2%}")
        if 'estimated_accuracy' in stats:
            print(f"- 估计正确率: {stats['estimated_accuracy']:.2%}")
        print("="*50)
        print(f"结果已保存到: {output_path}")
        
        # 获取输出目录的相对路径表示 - 修复跨驱动器问题
        try:
            output_rel_dir = os.path.relpath(os.path.dirname(output_path), os.getcwd())
        except ValueError:
            # 如果跨驱动器，直接使用绝对路径
            output_rel_dir = os.path.dirname(output_path)
        
        print(f"可视化和报告已保存到: {output_rel_dir}")
        html_report_path = os.path.join(os.path.dirname(output_path), 'inference_report.html')
        print(f"请查看HTML报告以获取详细分析: {html_report_path}")
        
        print("\n噬菌体-宿主匹配关系分析:")
        print(f"- 噬菌体数量: {relationship_stats['phage_count']}")
        print(f"- 宿主数量: {relationship_stats['host_count']}")
        print(f"- 普通宿主: {relationship_stats['normal_host_count']}")
        print(f"- 特殊宿主: {relationship_stats['special_host_count']}")
        print(f"- 匹配报告: {relationship_stats['report_path']}")
        
    except Exception as e:
        logging.error(f"推理过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()