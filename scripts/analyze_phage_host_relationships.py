#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析噬菌体-宿主关系
- 读取预测结果
- 分析噬菌体与宿主之间的关系
- 生成报告

使用方法：
python scripts/analyze_phage_host_relationships.py --input output/taxonomy_predictions.json --output output/relationship_analysis.html
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import csv

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
    parser = argparse.ArgumentParser(description='分析噬菌体-宿主关系')
    parser.add_argument('--input', type=str, required=True, help='预测结果文件路径')
    parser.add_argument('--output', type=str, required=True, help='分析报告输出路径')
    parser.add_argument('--threshold', type=float, default=0.6, help='匹配阈值')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    return parser.parse_args()

def load_predictions(predictions_path):
    """加载预测结果"""
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        logging.info(f"成功加载预测结果: {predictions_path}")
        logging.info(f"预测结果包含 {len(predictions)} 个实体")
        return predictions
    except Exception as e:
        logging.error(f"加载预测结果失败: {e}")
        raise

def analyze_relationships(predictions, threshold=0.6):
    """分析噬菌体-宿主关系"""
    # 分离噬菌体和宿主
    phages = {}
    hosts = {}
    for entity_id, result in predictions.items():
        if result['type'] == 'phage':
            phages[entity_id] = result
        elif result['type'] == 'host':
            hosts[entity_id] = result
    
    logging.info(f"找到 {len(phages)} 个噬菌体和 {len(hosts)} 个宿主")
    
    # 按预测类别分类宿主
    hosts_by_class = defaultdict(list)
    for host_id, host in hosts.items():
        prediction = host['prediction']
        hosts_by_class[prediction].append((host_id, host))
    
    # 找出特殊宿主（类别2）
    special_hosts = []
    if 2 in hosts_by_class:
        special_hosts = hosts_by_class[2]
        logging.info(f"找到 {len(special_hosts)} 个特殊宿主")
    
    # 为每个噬菌体找出最佳匹配的宿主
    matches = []
    for phage_id, phage in phages.items():
        # 寻找匹配的宿主
        best_matches = find_matching_hosts(phage_id, phage, hosts_by_class, threshold)
        matches.append({
            'phage_id': phage_id,
            'phage_confidence': phage['prediction_confidence'],
            'matches': best_matches
        })
    
    # 统计匹配情况
    matched_phages = sum(1 for m in matches if m['matches'])
    match_ratio = matched_phages / len(phages) if phages else 0
    
    # 统计结果
    stats = {
        'phage_count': len(phages),
        'host_count': len(hosts),
        'special_host_count': len(special_hosts),
        'matched_phages': matched_phages,
        'match_ratio': match_ratio,
        'host_types': {k: len(v) for k, v in hosts_by_class.items()}
    }
    
    logging.info(f"匹配统计: {stats}")
    
    return matches, special_hosts, stats

def find_matching_hosts(phage_id, phage, hosts_by_class, threshold):
    """为噬菌体找出匹配的宿主"""
    # 匹配策略:
    # 1. 基于ID相似性
    # 2. 可以扩展为基于功能或其他匹配逻辑
    matching_hosts = []
    
    # 主要从类别1的宿主中寻找匹配
    if 1 in hosts_by_class:
        for host_id, host in hosts_by_class[1]:
            # ID相似性算法
            similarity = calculate_similarity(phage_id, host_id)
            
            if similarity > threshold:
                matching_hosts.append({
                    'host_id': host_id,
                    'similarity': similarity,
                    'confidence': host['prediction_confidence']
                })
    
    # 按相似度排序
    matching_hosts.sort(key=lambda x: x['similarity'], reverse=True)
    return matching_hosts[:3]  # 返回前三个最佳匹配

def calculate_similarity(id1, id2):
    """计算两个ID之间的相似度"""
    # 这里可以实现更复杂的相似度算法
    # 简单示例：计算共同字符的比例
    id1, id2 = str(id1), str(id2)
    common = sum(1 for c in id1 if c in id2)
    return common / max(len(id1), len(id2))

def generate_report(matches, special_hosts, stats, output_path):
    """生成分析报告"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成CSV文件
    csv_path = os.path.join(output_dir, "phage_host_matches.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['噬菌体ID', '预测置信度', '匹配宿主ID', '相似度', '宿主置信度'])
        
        for match in matches:
            phage_id = match['phage_id']
            phage_conf = match['phage_confidence']
            
            if match['matches']:
                for host_match in match['matches']:
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
    special_csv_path = os.path.join(output_dir, "special_hosts.csv")
    with open(special_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['特殊宿主ID', '预测置信度', '备注'])
        
        for host_id, host in special_hosts:
            writer.writerow([
                host_id,
                host['prediction_confidence'],
                "特殊类别宿主"
            ])
    
    # 生成HTML报告
    html_content = generate_html_report(matches, special_hosts, stats)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"分析报告已保存到: {output_path}")
    logging.info(f"匹配结果CSV已保存到: {csv_path}")
    logging.info(f"特殊宿主CSV已保存到: {special_csv_path}")
    
    return {
        'report_path': output_path,
        'matches_csv': csv_path,
        'special_hosts_csv': special_csv_path
    }

def generate_html_report(matches, special_hosts, stats):
    """生成HTML格式的报告"""
    # 匹配表格HTML
    matches_table = ""
    for match in matches:
        phage_id = match['phage_id']
        phage_conf = match['phage_confidence']
        
        if match['matches']:
            for host_match in match['matches']:
                host_id = host_match['host_id']
                similarity = host_match['similarity']
                host_conf = host_match['confidence']
                
                matches_table += f"""
                <tr>
                    <td>{phage_id}</td>
                    <td>{phage_conf:.4f}</td>
                    <td>{host_id}</td>
                    <td>{similarity:.4f}</td>
                    <td>{host_conf:.4f}</td>
                </tr>
                """
        else:
            matches_table += f"""
            <tr>
                <td>{phage_id}</td>
                <td>{phage_conf:.4f}</td>
                <td colspan="3" class="no-match">无匹配宿主</td>
            </tr>
            """
    
    # 特殊宿主表格HTML
    special_hosts_table = ""
    for host_id, host in special_hosts:
        special_hosts_table += f"""
        <tr>
            <td>{host_id}</td>
            <td>{host['prediction_confidence']:.4f}</td>
            <td>特殊类别宿主</td>
        </tr>
        """
    
    # 完整HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>噬菌体-宿主关系分析</title>
        <meta charset="utf-8">
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
            <h1>噬菌体-宿主关系分析报告</h1>
            
            <div class="summary">
                <h2>总体统计</h2>
                <p>噬菌体总数: <strong>{stats['phage_count']}</strong></p>
                <p>宿主总数: <strong>{stats['host_count']}</strong></p>
                <p>特殊宿主数: <strong>{stats['special_host_count']}</strong></p>
                <p>成功匹配的噬菌体数: <strong>{stats['matched_phages']}</strong> ({stats['match_ratio']:.2%})</p>
                <p>宿主类型分布:
                    <ul>
                        {' '.join(f'<li>类别 {k}: {v}个</li>' for k, v in stats['host_types'].items())}
                    </ul>
                </p>
            </div>
            
            <h2>噬菌体-宿主匹配</h2>
            <table>
                <thead>
                    <tr>
                        <th>噬菌体ID</th>
                        <th>预测置信度</th>
                        <th>匹配宿主ID</th>
                        <th>相似度</th>
                        <th>宿主置信度</th>
                    </tr>
                </thead>
                <tbody>
                    {matches_table}
                </tbody>
            </table>
            
            <h2>特殊宿主</h2>
            <table>
                <thead>
                    <tr>
                        <th>宿主ID</th>
                        <th>预测置信度</th>
                        <th>备注</th>
                    </tr>
                </thead>
                <tbody>
                    {special_hosts_table}
                </tbody>
            </table>
            
            <h2>方法说明</h2>
            <p>本分析使用基于ID相似度的方法来匹配噬菌体和宿主。在实际应用中，应考虑使用更复杂的生物学相似性算法。</p>
            <p>特殊宿主指的是被分类为第3类(类别2)的宿主，这些宿主可能具有特殊的分类学特征或功能性特点。</p>
            
            <footer>
                <p>报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    return html

def create_visualizations(matches, special_hosts, stats, output_dir):
    """创建可视化图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 饼图：宿主类型分布
    plt.figure(figsize=(10, 6))
    host_types = stats['host_types']
    labels = [f'类别 {k}' for k in host_types.keys()]
    sizes = list(host_types.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('宿主类型分布')
    plt.savefig(os.path.join(output_dir, 'host_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 条形图：匹配情况
    plt.figure(figsize=(10, 6))
    match_counts = [stats['matched_phages'], stats['phage_count'] - stats['matched_phages']]
    plt.bar(['已匹配', '未匹配'], match_counts, color=['green', 'red'])
    plt.ylabel('噬菌体数量')
    plt.title('噬菌体匹配情况')
    plt.savefig(os.path.join(output_dir, 'phage_match_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 直方图：相似度分布
    similarities = []
    for match in matches:
        for host_match in match['matches']:
            similarities.append(host_match['similarity'])
    
    if similarities:
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, alpha=0.7, color='blue')
        plt.xlabel('相似度')
        plt.ylabel('频率')
        plt.title('匹配相似度分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"可视化图表已保存到: {output_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 1. 加载预测结果
        predictions = load_predictions(args.input)
        
        # 2. 分析噬菌体-宿主关系
        matches, special_hosts, stats = analyze_relationships(predictions, args.threshold)
        
        # 3. 生成报告
        output_files = generate_report(matches, special_hosts, stats, args.output)
        
        # 4. 生成可视化图表
        if args.visualize:
            output_dir = os.path.dirname(args.output)
            create_visualizations(matches, special_hosts, stats, output_dir)
        
        # 5. 打印摘要
        print("\n" + "="*50)
        print("噬菌体-宿主关系分析完成！")
        print(f"- 噬菌体总数: {stats['phage_count']}")
        print(f"- 宿主总数: {stats['host_count']}")
        print(f"- 特殊宿主数: {stats['special_host_count']}")
        print(f"- 匹配成功率: {stats['match_ratio']:.2%}")
        print("="*50)
        print(f"报告已保存到: {args.output}")
        print(f"匹配详情: {output_files['matches_csv']}")
        print(f"特殊宿主: {output_files['special_hosts_csv']}")
        
    except Exception as e:
        logging.error(f"分析过程出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 