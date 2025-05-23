# 模型版本说明

## 当前最佳模型（高性能GPU版本 v1.0）
- 文件名：best_graphsage_model.py
- 位置：models/
- 特点：
  - 增强的特征工程（节点度和影响力作为额外特征）
  - 使用改进的SE注意力机制
  - 3层GraphSAGE结构
  - 隐藏层维度128
  - 输入投影层和输出投影层
  - 多尺度残差连接和批量归一化
  - dropout率0.3
  - 支持混合精度训练
  - 使用AdamW优化器和余弦退火学习率
  - Kaiming权重初始化
  - **训练日期**: 2025-04-13

## 存档最佳模型
- 文件名：best_graphsage_model.py (旧版)
- 位置：models/
- 特点：
  - 使用SE注意力机制
  - 3层GraphSAGE结构
  - 隐藏层维度128
  - 包含残差连接和批量归一化
  - dropout率0.2

## 存档模型
- 文件名：graphsage_model.py
- 位置：archive/models/
- 说明：早期版本的GraphSAGE模型实现

## 模型权重文件
- best_graphsage_model.pt：高性能GPU优化版本权重（**黄金标准**）
- graphsage_best_attempt_2.pt：第二次尝试的最佳模型权重
- graphsage_best_attempt_1.pt：第一次尝试的最佳模型权重
- graphsage_model.pt：基础模型权重
- best_model.pt：早期最佳模型权重

## 训练脚本
- 当前最佳：train_best_graphsage.py (training/)
- 存档版本：train_graphsage.py (archive/training/)

## 性能指标
### 高性能GPU版本（v1.0实际指标）
- 训练集准确率：99.94%
- 验证集准确率：97.69%
- 测试集准确率：96.15%
- 训练集F1分数：0.9994
- 验证集F1分数：0.9769
- 测试集F1分数：0.9616
- 训练轮次：139（早停）
- GPU：NVIDIA GeForce RTX 2060
- GPU内存使用：22.49 MB

### 旧版模型
- 训练准确率：81.39%
- 验证准确率：80.21%
- 测试准确率：74.62%

## 未来优化方向
1. **减少模型复杂度**:
   - 尝试减少隐藏层维度到64或32
   - 考虑移除某些组件如注意力机制，评估对性能的影响
   - 优化网络结构以提高推理速度

2. **扩展到更大规模图**:
   - 添加图采样策略以处理更大规模图
   - 实现分布式训练支持
   - 优化内存使用以处理百万级节点图

3. **其他图卷积操作**:
   - 尝试GAT（图注意力网络）
   - 实现GIN（图同构网络）
   - 探索更新颖的图神经网络架构 

## ������ģ�ͣ�������
- �ļ�����light_graphsage_model.py
- λ�ã�models/
- �ص㣺
  - 2��GraphSAGE�ṹ
  - ���ز�ά��64
  - �Ƴ�ע��������
  - ��������һ���Ͳв�����
  - dropout��0.2
  - ��������94.69%
  - �����ٶ�����1.57x
  - ����׼ȷ�ʣ�0.8282
  - **ѵ������**: 2025-04-13
