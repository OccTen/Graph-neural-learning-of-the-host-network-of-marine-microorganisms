# 基于图神经网络的分类学特征推断研究

本项目利用GraphSAGE图神经网络实现了噬菌体与宿主之间的互作机制分析和可视化。

## 主要功能

- 通过GraphSAGE模型进行分类学特征推断
- 噬菌体-宿主互作网络可视化
- 特征空间降维分析和可视化
- 分类学分布统计

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

训练模型:
```bash
python scripts/train_taxonomy_improved.py --input data/features.csv --output output/model
```

可视化分析:
```bash
python scripts/visualize_phage_host_interactions.py --input output/taxonomy_predictions.json --output output/interaction_analysis --interactive
```
```

3. 创建requirements.txt文件：

在项目根目录创建requirements.txt文件，列出所有依赖库：
