#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphSAGE模型训练和评估的主入口
"""

import os
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GraphSAGE模型训练和评估')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--data', type=str, default='data/graph_data.pt', help='数据文件路径')
    parser.add_argument('--model', type=str, default='weights/best_graphsage_model.pt', help='模型权重文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--hidden', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--layers', type=int, default=3, help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    return parser.parse_args()

def ensure_dirs():
    """确保必要的目录存在"""
    dirs = ['data', 'weights', 'results', 'models', 'utils', 'training']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        
def main():
    """主函数"""
    args = parse_args()
    ensure_dirs()
    
    if args.train:
        logging.info("启动模型训练...")
        from training import train_main
        train_main()
    
    if args.evaluate:
        logging.info("启动模型评估...")
        try:
            from utils.evaluation import main as eval_main
            eval_main()
        except AttributeError:
            logging.error("评估模块缺少main函数，请确认utils/evaluation.py文件是否包含main函数")
        except ImportError:
            logging.error("无法导入评估模块，请确认utils/evaluation.py文件是否存在")
        
    if args.visualize:
        logging.info("生成可视化结果...")
        # 可视化代码...
    
    if not (args.train or args.evaluate or args.visualize):
        logging.info("请至少指定一个操作: --train, --evaluate 或 --visualize")
        print("使用方法: python main.py [--train] [--evaluate] [--visualize]")
        
if __name__ == "__main__":
    main() 