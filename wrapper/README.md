# 分类学特征推断系统 - 运行脚本

本目录包含用于简化系统运行的批处理脚本。这些脚本会自动设置正确的工作目录和路径，避免因路径问题导致的运行错误。

## 可用脚本

1. **run_inference.bat** - 运行分类学特征推断
   - 自动加载模型、测试数据和编码器
   - 结果保存到`output/taxonomy_predictions.json`

2. **run_analysis.bat** - 分析噬菌体-宿主关系
   - 读取推断结果生成详细分析报告
   - 创建可视化图表
   - 结果保存到`output/relationship_analysis.html`

## 使用方法

### 运行推断

1. 双击`run_inference.bat`
2. 等待推断完成
3. 结果将保存到`output`目录

### 运行分析

1. 确保已完成推断并生成了`taxonomy_predictions.json`
2. 双击`run_analysis.bat`
3. 分析报告将自动生成并保存

## 常见问题

### 如果脚本无法运行

- 确保已安装Python和所需依赖
- 检查`models`目录中是否存在`best_graphsage_model.py`
- 确保目录结构正确（脚本位于`wrapper`目录内）

### 如果需要修改参数

可以编辑批处理文件中的以下参数：

```batch
rem 在run_inference.bat中
set MODEL_PATH=weights\taxonomy_graphsage_optimized.pt
set TEST_DATA=data\Independent test set.xlsx
set ENCODERS=data\taxonomy_values.json
set OUTPUT=output\taxonomy_predictions.json

rem 在run_analysis.bat中
set INPUT=output\taxonomy_predictions.json
set OUTPUT=output\relationship_analysis.html
``` 