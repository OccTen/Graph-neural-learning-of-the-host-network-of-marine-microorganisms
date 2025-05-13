"""
网络构建模块，用于构建海洋微生物宿主网络
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from typing import Dict, List, Tuple, Optional, Set
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import DATA_DIR, MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkBuilder:
    """网络构建类，用于构建海洋微生物宿主网络"""
    
    def __init__(self, config: Dict = None):
        """
        初始化网络构建器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or MODEL_CONFIG
        self.scaler = StandardScaler()
        
    def _extract_node_features(self, node_data: pd.DataFrame) -> np.ndarray:
        """
        提取节点特征
        
        Args:
            node_data: 节点数据DataFrame
            
        Returns:
            节点特征矩阵
        """
        try:
            # 1. 基本特征
            features = []
            for _, row in node_data.iterrows():
                # 序列特征
                sequence = row['sequence']
                gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
                at_content = (sequence.count('A') + sequence.count('T')) / len(sequence)
                
                # 结构特征
                structure = row['structure']
                stem_count = structure.count('(') + structure.count(')')
                loop_count = structure.count('.')
                
                # 功能特征
                function = row['function']
                is_enzyme = 1 if 'enzyme' in function.lower() else 0
                is_transporter = 1 if 'transporter' in function.lower() else 0
                
                # 组合特征
                node_features = [
                    gc_content,
                    at_content,
                    stem_count / len(structure),
                    loop_count / len(structure),
                    is_enzyme,
                    is_transporter,
                    len(sequence),
                    len(structure)
                ]
                features.append(node_features)
            
            # 标准化特征
            features = np.array(features)
            features = self.scaler.fit_transform(features)
            
            return features
        except Exception as e:
            logger.error(f"提取节点特征时出错: {str(e)}")
            raise
            
    def _extract_edge_features(self, edge_data: pd.DataFrame) -> np.ndarray:
        """
        提取边特征
        
        Args:
            edge_data: 边数据DataFrame
            
        Returns:
            边特征矩阵
        """
        try:
            features = []
            for _, row in edge_data.iterrows():
                # 交互特征
                interaction_type = row['interaction_type']
                interaction_strength = row['interaction_strength']
                
                # 时间特征
                time_point = row['time_point']
                duration = row['duration']
                
                # 环境特征
                temperature = row['temperature']
                ph = row['ph']
                salinity = row['salinity']
                
                # 组合特征
                edge_features = [
                    interaction_strength,
                    time_point,
                    duration,
                    temperature,
                    ph,
                    salinity,
                    1.0 if interaction_type == 'positive' else 0.0,
                    1.0 if interaction_type == 'negative' else 0.0
                ]
                features.append(edge_features)
            
            # 标准化特征
            features = np.array(features)
            features = self.scaler.fit_transform(features)
            
            return features
        except Exception as e:
            logger.error(f"提取边特征时出错: {str(e)}")
            raise
            
    def _augment_data(self, data: Data) -> Data:
        """
        数据增强
        
        Args:
            data: 原始图数据
            
        Returns:
            增强后的图数据
        """
        try:
            # 1. 节点特征增强
            x = data.x.numpy()
            noise = np.random.normal(0, 0.1, x.shape)
            x_aug = x + noise
            data.x = torch.FloatTensor(x_aug)
            
            # 2. 边特征增强
            edge_attr = data.edge_attr.numpy()
            noise = np.random.normal(0, 0.05, edge_attr.shape)
            edge_attr_aug = edge_attr + noise
            data.edge_attr = torch.FloatTensor(edge_attr_aug)
            
            # 3. 添加随机边
            num_nodes = data.x.size(0)
            num_new_edges = int(data.edge_index.size(1) * 0.1)  # 添加10%的新边
            
            new_edges = []
            for _ in range(num_new_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:
                    new_edges.append([src, dst])
            
            if new_edges:
                new_edges = torch.LongTensor(new_edges).t()
                data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
                
                # 为新边添加特征
                new_edge_attr = torch.FloatTensor(np.random.normal(0, 0.1, (len(new_edges), data.edge_attr.size(1))))
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
            
            return data
        except Exception as e:
            logger.error(f"数据增强时出错: {str(e)}")
            raise
            
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        加载预处理后的数据
        
        Returns:
            节点特征矩阵、边索引、边属性、节点映射
        """
        try:
            # 检查文件是否存在
            required_files = ['processed_data_train.csv', 'processed_data_val.csv', 'processed_data_test.csv']
            for file in required_files:
                file_path = os.path.join(DATA_DIR, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 加载训练数据
            train_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_data_train.csv'))
            val_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_data_val.csv'))
            test_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_data_test.csv'))
            
            # 合并所有数据以构建完整的图
            all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
            
            # 创建节点映射
            unique_nodes = pd.concat([all_data['Phage'], all_data['Host'], all_data['Non-host']]).unique()
            node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
            
            # 创建边索引
            edge_index = []
            edge_attr = []
            
            # 添加正样本边（Phage-Host）
            for _, row in all_data.iterrows():
                phage_idx = node_mapping[row['Phage']]
                host_idx = node_mapping[row['Host']]
                edge_index.append([phage_idx, host_idx])
                edge_attr.append([1.0])  # 正样本标签
                
                # 添加反向边
                edge_index.append([host_idx, phage_idx])
                edge_attr.append([1.0])
            
            # 添加负样本边（Phage-Non-host）
            for _, row in all_data.iterrows():
                phage_idx = node_mapping[row['Phage']]
                non_host_idx = node_mapping[row['Non-host']]
                edge_index.append([phage_idx, non_host_idx])
                edge_attr.append([0.0])  # 负样本标签
                
                # 添加反向边
                edge_index.append([non_host_idx, phage_idx])
                edge_attr.append([0.0])
            
            # 转换为numpy数组
            edge_index = np.array(edge_index).T
            edge_attr = np.array(edge_attr)
            
            # 创建节点特征矩阵
            num_nodes = len(node_mapping)
            num_features = 3  # 根据预处理后的特征维度
            node_features = np.zeros((num_nodes, num_features))
            
            # 保存节点映射
            with open(os.path.join(DATA_DIR, 'node_mapping.json'), 'w') as f:
                json.dump(node_mapping, f)
            
            # 保存数据
            np.save(os.path.join(DATA_DIR, 'node_features.npy'), node_features)
            np.save(os.path.join(DATA_DIR, 'edge_index.npy'), edge_index)
            np.save(os.path.join(DATA_DIR, 'edge_attr.npy'), edge_attr)
            
            logger.info(f"成功加载数据")
            logger.info(f"节点特征形状: {node_features.shape}")
            logger.info(f"边索引形状: {edge_index.shape}")
            logger.info(f"边属性形状: {edge_attr.shape}")
            logger.info(f"节点数量: {len(node_mapping)}")
            
            return node_features, edge_index, edge_attr, node_mapping
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
            
    def _build_neighbor_dict(self, edge_index: torch.Tensor) -> Dict[int, List[int]]:
        """
        构建邻居字典，提高采样效率
        
        Args:
            edge_index: 边索引
            
        Returns:
            邻居字典
        """
        neighbor_dict = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in neighbor_dict:
                neighbor_dict[src] = []
            if dst not in neighbor_dict:
                neighbor_dict[dst] = []
            neighbor_dict[src].append(dst)
            neighbor_dict[dst].append(src)
        return neighbor_dict

    def _sample_neighbors(self, node_idx: int, neighbor_dict: Dict[int, List[int]], 
                         num_samples: int = 5) -> Set[int]:
        """
        采样节点的邻居
        
        Args:
            node_idx: 目标节点索引
            neighbor_dict: 邻居字典
            num_samples: 采样数量
            
        Returns:
            采样到的邻居节点索引集合
        """
        try:
            # 获取所有邻居
            neighbors = neighbor_dict.get(node_idx, [])
            
            # 如果邻居数量不足，重复采样
            if len(neighbors) < num_samples:
                sampled = neighbors * (num_samples // len(neighbors) + 1)
                sampled = sampled[:num_samples]
            else:
                # 随机采样
                sampled = np.random.choice(neighbors, num_samples, replace=False)
            
            return set(sampled)
        except Exception as e:
            logger.error(f"邻居采样时出错: {str(e)}")
            raise
            
    def _aggregate_features_batch(self, node_indices: List[int], 
                                neighbor_dict: Dict[int, List[int]], 
                                x: torch.Tensor, 
                                num_samples: int = 5) -> torch.Tensor:
        """
        批量聚合特征
        
        Args:
            node_indices: 目标节点索引列表
            neighbor_dict: 邻居字典
            x: 节点特征矩阵
            num_samples: 采样数量
            
        Returns:
            聚合后的特征矩阵
        """
        try:
            batch_size = len(node_indices)
            feat_dim = x.size(1)
            aggregated = torch.zeros((batch_size, feat_dim * 2), device=x.device)
            
            for i, node_idx in enumerate(node_indices):
                # 获取目标节点特征
                target_feat = x[node_idx]
                
                # 采样邻居
                neighbors = self._sample_neighbors(node_idx, neighbor_dict, num_samples)
                
                # 获取邻居特征
                if neighbors:
                    neighbor_feats = x[list(neighbors)]
                    neighbor_mean = neighbor_feats.mean(dim=0)
                else:
                    neighbor_mean = torch.zeros_like(target_feat)
                
                # 拼接特征
                aggregated[i] = torch.cat([target_feat, neighbor_mean])
            
            return aggregated
        except Exception as e:
            logger.error(f"批量特征聚合时出错: {str(e)}")
            raise
            
    def prepare_graphsage_data(self, data: Data, num_samples: int = 5, 
                             batch_size: int = 256) -> Data:
        """
        准备GraphSAGE训练数据
        
        Args:
            data: 原始图数据
            num_samples: 每个节点的邻居采样数量
            batch_size: 批处理大小
            
        Returns:
            处理后的图数据
        """
        try:
            logger.info("开始准备GraphSAGE数据...")
            
            # 确保图是无向的
            edge_index = to_undirected(data.edge_index)
            logger.info("图已转换为无向图")
            
            # 构建邻居字典
            neighbor_dict = self._build_neighbor_dict(edge_index)
            logger.info("邻居字典构建完成")
            
            # 批量处理节点
            num_nodes = data.x.size(0)
            aggregated_features = []
            
            # 使用tqdm显示进度
            for i in tqdm(range(0, num_nodes, batch_size), desc="处理节点"):
                batch_indices = list(range(i, min(i + batch_size, num_nodes)))
                batch_features = self._aggregate_features_batch(
                    batch_indices, neighbor_dict, data.x, num_samples
                )
                aggregated_features.append(batch_features)
            
            # 合并所有批次的特征
            data.x = torch.cat(aggregated_features, dim=0)
            
            # 更新边索引
            data.edge_index = edge_index
            
            logger.info(f"GraphSAGE数据准备完成")
            logger.info(f"处理后的节点特征: {data.x.shape}")
            logger.info(f"处理后的边索引: {data.edge_index.shape}")
            
            return data
        except Exception as e:
            logger.error(f"准备GraphSAGE数据时出错: {str(e)}")
            raise
            
    def build_network(self, node_features: np.ndarray, edge_index: np.ndarray, 
                     edge_attr: np.ndarray) -> Data:
        """
        构建图网络
        
        Args:
            node_features: 节点特征矩阵
            edge_index: 边索引
            edge_attr: 边属性
            
        Returns:
            PyTorch Geometric Data对象
        """
        try:
            # 数据预处理
            # 1. 确保节点特征没有NaN值
            if np.isnan(node_features).any():
                logger.warning("节点特征中存在NaN值，使用均值填充")
                node_features = np.nan_to_num(node_features, nan=np.nanmean(node_features))
            
            # 2. 确保边索引在有效范围内
            max_node_idx = node_features.shape[0] - 1
            if edge_index.max() > max_node_idx or edge_index.min() < 0:
                raise ValueError(f"边索引超出范围: [{edge_index.min()}, {edge_index.max()}]")
            
            # 3. 确保边属性没有NaN值
            if np.isnan(edge_attr).any():
                logger.warning("边属性中存在NaN值，使用均值填充")
                edge_attr = np.nan_to_num(edge_attr, nan=np.nanmean(edge_attr))
            
            # 转换为PyTorch张量
            x = torch.FloatTensor(node_features)
            edge_index = torch.LongTensor(edge_index)
            edge_attr = torch.FloatTensor(edge_attr)
            
            # 创建Data对象
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=edge_attr  # 使用边属性作为标签
            )
            
            # 数据增强
            data = self._augment_data(data)
            
            # 准备GraphSAGE数据
            data = self.prepare_graphsage_data(data)
            
            logger.info(f"成功构建图网络")
            logger.info(f"节点特征: {data.x.shape}")
            logger.info(f"边索引: {data.edge_index.shape}")
            logger.info(f"边属性: {data.edge_attr.shape}")
            
            return data
        except Exception as e:
            logger.error(f"构建网络时出错: {str(e)}")
            raise
            
    def split_edges(self, data: Data, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Data:
        """
        分割边为训练集、验证集和测试集
        
        Args:
            data: 图数据
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            更新后的图数据
        """
        try:
            # 获取边索引和属性
            edge_index = data.edge_index.numpy()
            edge_attr = data.edge_attr.numpy()
            
            # 计算分割点
            num_edges = edge_index.shape[1]
            num_val = int(num_edges * val_ratio)
            num_test = int(num_edges * test_ratio)
            
            # 确保每个节点在训练集中至少有一条边
            node_degrees = np.zeros(data.x.shape[0])
            for i, j in edge_index.T:
                node_degrees[i] += 1
                node_degrees[j] += 1
            
            # 随机打乱边，但确保每个节点至少有一条边在训练集中
            indices = np.random.permutation(num_edges)
            train_indices = []
            val_indices = []
            test_indices = []
            
            # 首先为每个节点选择一条边到训练集
            node_covered = set()
            for idx in indices:
                i, j = edge_index[:, idx]
                if i not in node_covered or j not in node_covered:
                    train_indices.append(idx)
                    node_covered.add(i)
                    node_covered.add(j)
                    if len(train_indices) >= num_edges - num_val - num_test:
                        break
            
            # 剩余的边随机分配到验证集和测试集
            remaining_indices = [idx for idx in indices if idx not in train_indices]
            val_indices = remaining_indices[:num_val]
            test_indices = remaining_indices[num_val:num_val + num_test]
            train_indices.extend(remaining_indices[num_val + num_test:])
            
            # 创建掩码
            train_mask = np.zeros(num_edges, dtype=bool)
            val_mask = np.zeros(num_edges, dtype=bool)
            test_mask = np.zeros(num_edges, dtype=bool)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True
            
            # 更新Data对象
            data.train_mask = torch.BoolTensor(train_mask)
            data.val_mask = torch.BoolTensor(val_mask)
            data.test_mask = torch.BoolTensor(test_mask)
            
            logger.info(f"成功分割边")
            logger.info(f"训练集边数量: {sum(train_mask)}")
            logger.info(f"验证集边数量: {sum(val_mask)}")
            logger.info(f"测试集边数量: {sum(test_mask)}")
            
            return data
        except Exception as e:
            logger.error(f"分割边时出错: {str(e)}")
            raise
            
    def save_network(self, data: Data, save_path: str):
        """
        保存图网络
        
        Args:
            data: 图数据
            save_path: 保存路径
        """
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存数据
            torch.save(data, save_path)
            logger.info(f"图网络已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存网络时出错: {str(e)}")
            raise

def main():
    """主函数，用于测试网络构建器"""
    try:
        # 初始化网络构建器
        builder = NetworkBuilder()
        
        # 加载数据
        node_features, edge_index, edge_attr, node_mapping = builder.load_data()
        
        # 构建网络
        data = builder.build_network(node_features, edge_index, edge_attr)
        
        # 分割边
        data = builder.split_edges(data)
        
        # 保存网络
        save_path = os.path.join(DATA_DIR, 'graph_data.pt')
        builder.save_network(data, save_path)
        
        logger.info("网络构建完成")
        
    except Exception as e:
        logger.error(f"网络构建过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 