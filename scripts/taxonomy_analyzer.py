#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类学名称分析工具
- 分析分类学名称
- 生成分类学名称列表
- 创建分类学映射字典
"""

import os
import pandas as pd
import json
import logging
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('data', 'taxonomy_analysis.log'))
    ]
)

def load_excel_data(file_path):
    """加载Excel数据，包括所有工作表"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            # 尝试其他可能的位置
            alt_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'Training set.xlsx'))
            if os.path.exists(alt_path):
                logging.info(f"找到替代路径: {alt_path}")
                file_path = alt_path
            else:
                # 尝试使用相对路径
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.lower() == 'training set.xlsx':
                            found_path = os.path.join(root, file)
                            logging.info(f"找到文件: {found_path}")
                            file_path = found_path
                            break
                    if file_path != alt_path:
                        break
                
                if not os.path.exists(file_path):
                    # 尝试在M:\4.9\data中查找
                    try:
                        base_path = r"M:\4.9\data"
                        if os.path.exists(os.path.join(base_path, 'Training set.xlsx')):
                            file_path = os.path.join(base_path, 'Training set.xlsx')
                            logging.info(f"找到文件: {file_path}")
                    except:
                        pass
                    
            if not os.path.exists(file_path):
                logging.error("在多个位置尝试后仍未找到文件，退出")
                return None, None
        
        logging.info(f"正在加载文件: {file_path}")
        
        # 获取所有工作表名称
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        logging.info(f"Excel文件 {file_path} 包含 {len(sheet_names)} 个工作表: {sheet_names}")
        
        # 读取第一个工作表作为示例
        first_sheet = pd.read_excel(file_path, sheet_name=0)
        logging.info(f"第一个工作表 '{sheet_names[0]}' 形状: {first_sheet.shape}")
        
        # 加载所有工作表
        all_sheets = {}
        for sheet_name in sheet_names:
            sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
            all_sheets[sheet_name] = sheet_data
            logging.info(f"工作表 '{sheet_name}' 形状: {sheet_data.shape}")
        
        return first_sheet, all_sheets
    except Exception as e:
        logging.error(f"加载Excel数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def analyze_taxonomy_columns(df):
    """分析第一行，确定分类学列的结构"""
    # 检查第一行是否包含列名信息
    first_row = df.iloc[0].tolist()
    column_names = df.columns.tolist()
    
    logging.info(f"列名: {column_names}")
    logging.info(f"第一行: {first_row}")
    
    # 识别分类学相关列
    taxonomy_columns = {}
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    entities = ['Phage', 'Host', 'Non-host']
    
    # 检查第一行是否包含分类学级别
    has_taxonomy_header = False
    if 'kingdom' in first_row:
        has_taxonomy_header = True
        logging.info("检测到第一行包含分类学级别信息")
    
    if has_taxonomy_header:
        # 如果第一行包含级别信息，使用它来映射列
        for entity in entities:
            entity_col_idx = None
            for i, col in enumerate(column_names):
                if col == entity:
                    entity_col_idx = i
                    break
            
            if entity_col_idx is not None:
                taxonomy_columns[entity] = {'accession': entity_col_idx}
                
                # 寻找该实体相关的分类列
                for level_idx, level in enumerate(taxonomy_levels):
                    for i in range(entity_col_idx + 1, len(column_names)):
                        if i < len(first_row) and first_row[i] == level:
                            taxonomy_columns[entity][level] = i
                            break
    else:
        # 如果第一行不是清晰的级别信息，使用基于位置的推断
        for entity_idx, entity in enumerate(entities):
            entity_col_idx = column_names.index(entity) if entity in column_names else None
            if entity_col_idx is not None:
                taxonomy_columns[entity] = {'accession': entity_col_idx}
                
                # 推断分类学列，基于位置
                base_idx = entity_col_idx + 1
                for level_idx, level in enumerate(taxonomy_levels):
                    if base_idx + level_idx < len(column_names):
                        taxonomy_columns[entity][level] = base_idx + level_idx
    
    logging.info(f"识别的分类学列: {taxonomy_columns}")
    return taxonomy_columns, has_taxonomy_header

def extract_taxonomy_values(df, taxonomy_columns, has_taxonomy_header):
    """提取所有的分类学值"""
    taxonomy_values = defaultdict(lambda: defaultdict(set))
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    
    # 确定数据起始行
    start_row = 1 if has_taxonomy_header else 0
    
    # 对于每个实体
    for entity, columns in taxonomy_columns.items():
        for level in taxonomy_levels:
            if level in columns:
                col_idx = columns[level]
                for i in range(start_row, len(df)):
                    value = df.iloc[i, col_idx]
                    if pd.notna(value) and value != '':
                        taxonomy_values[entity][level].add(value)
    
    # 转换为普通字典并排序
    result = {}
    for entity, levels in taxonomy_values.items():
        result[entity] = {}
        for level, values in levels.items():
            result[entity][level] = sorted(list(values))
    
    # 打印统计信息
    for entity, levels in result.items():
        logging.info(f"\n{entity} 分类学级别:")
        for level, values in levels.items():
            logging.info(f"  - {level}: {len(values)} 个唯一值")
            logging.info(f"    示例: {values[:5]}")
    
    return result

def create_taxonomy_mapping(taxonomy_values):
    """创建分类学映射字典"""
    mapping = {}
    entity_codes = {'Phage': 'P', 'Host': 'H', 'Non-host': 'N'}
    level_codes = {'kingdom': 'K', 'phylum': 'P', 'class': 'C', 'order': 'O', 'family': 'F', 'genus': 'G'}
    
    node_id = 0
    
    # 为每个实体和分类级别创建映射
    for entity, levels in taxonomy_values.items():
        for level, values in levels.items():
            for value in values:
                # 创建唯一的标识符
                entity_code = entity_codes.get(entity, entity[0])
                level_code = level_codes.get(level, level[0])
                key = f"{entity_code}_{level_code}_{value}"
                
                mapping[key] = {
                    'node_id': node_id,
                    'entity': entity,
                    'level': level,
                    'value': value
                }
                node_id += 1
    
    logging.info(f"创建了分类学映射字典，包含 {len(mapping)} 个条目")
    return mapping

def save_taxonomy_data(taxonomy_values, taxonomy_mapping, output_dir='data'):
    """保存分类学数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分类学值
    values_path = os.path.join(output_dir, 'taxonomy_values.json')
    with open(values_path, 'w') as f:
        json.dump(taxonomy_values, f, indent=2)
    logging.info(f"分类学值已保存至: {values_path}")
    
    # 保存分类学映射
    mapping_path = os.path.join(output_dir, 'taxonomy_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(taxonomy_mapping, f, indent=2)
    logging.info(f"分类学映射已保存至: {mapping_path}")
    
    # 创建更简洁的映射版本，用于模型训练
    simple_mapping = {}
    for key, value in taxonomy_mapping.items():
        simple_mapping[key] = value['node_id']
    
    simple_path = os.path.join(output_dir, 'taxonomy_node_mapping.json')
    with open(simple_path, 'w') as f:
        json.dump(simple_mapping, f, indent=2)
    logging.info(f"简化的分类学映射已保存至: {simple_path}")
    
    return values_path, mapping_path, simple_path

def analyze_cross_references(df, taxonomy_columns, has_taxonomy_header):
    """分析交叉引用，检查同一行中的不同实体是否共享分类"""
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    entities = ['Phage', 'Host', 'Non-host']
    
    # 确定数据起始行
    start_row = 1 if has_taxonomy_header else 0
    
    # 统计每个级别的交叉引用
    cross_refs = defaultdict(Counter)
    
    # 对于每一行数据
    for i in range(start_row, len(df)):
        # 对于每个分类级别
        for level in taxonomy_levels:
            # 收集该行中所有实体在此级别的值
            level_values = {}
            for entity in entities:
                if entity in taxonomy_columns and level in taxonomy_columns[entity]:
                    col_idx = taxonomy_columns[entity][level]
                    value = df.iloc[i, col_idx]
                    if pd.notna(value) and value != '':
                        level_values[entity] = value
            
            # 检查是否有交叉引用
            if len(level_values) > 1:
                for entity1 in entities:
                    for entity2 in entities:
                        if entity1 != entity2 and entity1 in level_values and entity2 in level_values:
                            # 记录交叉引用
                            key = f"{entity1}-{entity2}"
                            value_pair = (level_values[entity1], level_values[entity2])
                            cross_refs[f"{level}:{key}"][value_pair] += 1
    
    # 输出交叉引用统计
    logging.info("\n分类学交叉引用分析:")
    for key, counts in cross_refs.items():
        level, entity_pair = key.split(':', 1)
        logging.info(f"  - {level} 级别 ({entity_pair}):")
        
        # 只显示前5个最常见的交叉引用
        for (val1, val2), count in counts.most_common(5):
            logging.info(f"    * {val1} - {val2}: {count} 次")
    
    return cross_refs

def main():
    """主函数"""
    logging.info("开始分类学名称分析...")
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    
    # 尝试列出当前工作目录的内容
    try:
        cwd = os.getcwd()
        logging.info(f"当前工作目录: {cwd}")
        contents = os.listdir(cwd)
        logging.info(f"目录内容: {contents}")
        
        # 查看data目录是否存在及其内容
        data_dir = os.path.join(cwd, 'data')
        if os.path.exists(data_dir):
            data_contents = os.listdir(data_dir)
            logging.info(f"data目录内容: {data_contents}")
            
            # 查找Training set.xlsx
            excel_files = [f for f in data_contents if f.lower().endswith('.xlsx')]
            logging.info(f"找到的Excel文件: {excel_files}")
    except Exception as e:
        logging.error(f"列出目录内容时出错: {str(e)}")
    
    # 加载Excel数据 - 尝试多个可能的路径
    logging.info("正在加载Excel数据...")
    file_paths = [
        os.path.join('data', 'Training set.xlsx'),  # 相对于当前目录
        os.path.abspath(os.path.join(cwd, 'data', 'Training set.xlsx')),  # 绝对路径
        r"M:\4.9\data\Training set.xlsx"  # 指定驱动器路径
    ]
    
    first_sheet = None
    all_sheets = None
    
    for path in file_paths:
        logging.info(f"尝试加载: {path}")
        first_sheet, all_sheets = load_excel_data(path)
        if first_sheet is not None:
            logging.info(f"成功加载: {path}")
            break
    
    if first_sheet is None:
        logging.error("无法加载Excel数据，退出")
        return
    
    # 分析分类学列
    logging.info("正在分析分类学列...")
    taxonomy_columns, has_taxonomy_header = analyze_taxonomy_columns(first_sheet)
    
    # 提取分类学值
    logging.info("正在提取分类学值...")
    taxonomy_values = extract_taxonomy_values(first_sheet, taxonomy_columns, has_taxonomy_header)
    
    # 分析交叉引用
    logging.info("正在分析交叉引用...")
    cross_refs = analyze_cross_references(first_sheet, taxonomy_columns, has_taxonomy_header)
    
    # 创建分类学映射
    logging.info("正在创建分类学映射...")
    taxonomy_mapping = create_taxonomy_mapping(taxonomy_values)
    
    # 保存分类学数据
    logging.info("正在保存分类学数据...")
    values_path, mapping_path, simple_path = save_taxonomy_data(taxonomy_values, taxonomy_mapping)
    
    # 查看所有工作表中的分类学信息
    if all_sheets:
        logging.info("\n分析所有工作表的分类学信息:")
        for sheet_name, sheet_data in all_sheets.items():
            logging.info(f"\n工作表: {sheet_name}")
            sheet_columns, sheet_has_header = analyze_taxonomy_columns(sheet_data)
            sheet_values = extract_taxonomy_values(sheet_data, sheet_columns, sheet_has_header)
    
    logging.info("分类学名称分析完成!")

if __name__ == "__main__":
    main() 