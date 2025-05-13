#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化启动器 - 简化版
自动处理路径问题并启动可视化分析
"""

import os
import sys
import subprocess
import logging
import argparse
import glob

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_and_install_dependencies():
    """检查并安装必要的库"""
    required_packages = [
        "matplotlib",
        "seaborn",
        "networkx",
        "scikit-learn",
        "pyvis"
    ]
    
    missing_packages = []
    
    # 检查每个库是否已安装
    for package in required_packages:
        try:
            __import__(package)
            logging.info(f"√ {package} 已安装")
        except ImportError:
            logging.warning(f"× {package} 未安装")
            missing_packages.append(package)
    
    # 如果有缺失的库，安装它们
    if missing_packages:
        print("\n正在安装缺少的库...")
        for package in missing_packages:
            print(f"安装 {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"√ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"× {package} 安装失败: {str(e)}")
                return False
        
        print("所有缺少的库已安装完成")
    
    return True

def find_prediction_file(project_root):
    """查找可用的预测结果文件"""
    # 尝试默认路径
    default_path = os.path.join(project_root, "output", "taxonomy_predictions.json")
    if os.path.exists(default_path):
        return default_path
        
    # 尝试查找其他json文件
    prediction_files = glob.glob(os.path.join(project_root, "output", "*.json"))
    if prediction_files:
        # 按修改时间排序，取最新的
        prediction_files.sort(key=os.path.getmtime, reverse=True)
        print(f"找到 {len(prediction_files)} 个可能的预测文件，使用最新的: {os.path.basename(prediction_files[0])}")
        return prediction_files[0]
    
    # 没有找到任何文件
    return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="微生物-宿主互作可视化分析启动器")
    parser.add_argument('--input', type=str, help='输入JSON文件路径（taxonomy_predictions.json）')
    parser.add_argument('--output', type=str, help='输出目录路径')
    parser.add_argument('--fontsize', type=int, default=16, help='图表字体大小(默认16)')
    parser.add_argument('--dpi', type=int, default=400, help='图表DPI(默认400)')
    parser.add_argument('--no_interactive', action='store_true', help='不生成交互式可视化')
    return parser.parse_args()

def main():
    """主函数"""
    print("\n=== 微生物与宿主互作可视化分析启动器 ===\n")
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查依赖
    print("检查必要的Python库...")
    if not check_and_install_dependencies():
        print("无法安装所有必要的库，请手动安装以下库:")
        print("pip install matplotlib seaborn networkx scikit-learn pyvis")
        return 1
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 设置输入输出路径
    if args.input:
        input_file = args.input
    else:
        # 自动查找预测文件
        input_file = find_prediction_file(project_root)
        if not input_file:
            print("错误: 无法找到预测结果文件")
            print("请手动指定输入文件路径，或先运行模型推理脚本生成预测结果文件")
            return 1
    
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(project_root, "output", "interaction_analysis")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        alt_file = os.path.join(project_root, "output", "latest_taxonomy_predictions.json")
        if os.path.exists(alt_file):
            print(f"找到替代文件: {alt_file}")
            input_file = alt_file
        else:
            print("请先运行模型推理脚本生成预测结果文件")
            return 1
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 构建可视化脚本路径
    vis_script = os.path.join(project_root, "scripts", "visualize_phage_host_interactions.py")
    
    if not os.path.exists(vis_script):
        print(f"错误: 可视化脚本不存在: {vis_script}")
        return 1
    
    # 构建并运行命令
    cmd = [
        sys.executable,  # 当前Python解释器
        vis_script,
        "--input", input_file,
        "--output", output_dir,
        "--fontsize", str(args.fontsize),
        "--dpi", str(args.dpi)
    ]
    
    # 添加交互式参数
    if not args.no_interactive:
        cmd.append("--interactive")
    
    print("\n开始运行可视化分析...")
    print(f"- 输入文件: {input_file}")
    print(f"- 输出目录: {output_dir}")
    print(f"- 可视化脚本: {vis_script}")
    print(f"- 字体大小: {args.fontsize}")
    print(f"- 图像DPI: {args.dpi}")
    print(f"- 交互式可视化: {'否' if args.no_interactive else '是'}")
    print("\n正在处理，请稍候...\n")
    
    try:
        # 运行命令
        process = subprocess.run(cmd, text=True, capture_output=True)
        
        # 检查是否成功
        if process.returncode == 0:
            print("\n√ 可视化分析完成!")
            print(f"\n生成的可视化结果位于: {output_dir}")
            
            # 尝试打开输出目录
            try:
                os.startfile(output_dir)
                print("已自动打开输出目录")
            except:
                print(f"请手动打开输出目录查看结果: {output_dir}")
            
            return 0
        else:
            print("\n× 可视化分析失败")
            print("\n错误信息:")
            print(process.stderr)
            return 1
            
    except Exception as e:
        print(f"\n× 运行时出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 