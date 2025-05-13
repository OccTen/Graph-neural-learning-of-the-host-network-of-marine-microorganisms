#!/usr/bin/env python3
"""
分类学图GraphSAGE模型训练启动器
此脚本直接在正确的路径中启动训练
"""
import os
import sys

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 添加项目根目录到路径
sys.path.insert(0, project_root)

# 直接指定导入路径
training_path = os.path.join(project_root, "training")
sys.path.insert(0, training_path)

try:
    # 直接从training目录导入
    from train_best_graphsage import main
    print(f"成功从 {training_path} 导入main函数")
except ImportError as e:
    print(f"导入错误: {str(e)}")
    print(f"当前Python路径: {sys.path}")
    print(f"尝试从 {training_path} 导入train_best_graphsage.py失败")
    print(f"请确认文件 {os.path.join(training_path, 'train_best_graphsage.py')} 存在")
    # 尝试列出目录内容
    try:
        print(f"目录 {training_path} 的内容:")
        for file in os.listdir(training_path):
            print(f"  - {file}")
    except Exception as e:
        print(f"无法列出目录内容: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    print("开始训练分类学图GraphSAGE模型...")
    try:
        main()
        print("训练完成!")
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc() 