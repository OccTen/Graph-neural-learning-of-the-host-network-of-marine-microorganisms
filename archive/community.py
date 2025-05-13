"""
社区检测模块，实现Louvain社区检测算法
"""

import numpy as np
import networkx as nx
from typing import Dict, Set, List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LouvainCommunityDetection:
    """Louvain社区检测算法实现"""
    
    def __init__(self, resolution: float = 1.0, random_state: int = None):
        """
        初始化Louvain社区检测器
        
        Args:
            resolution: 分辨率参数，控制社区大小
            random_state: 随机种子
        """
        self.resolution = resolution
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
    def _compute_modularity(self, G: nx.Graph, communities: Dict) -> float:
        """
        计算模块度
        
        Args:
            G: 网络图
            communities: 社区分配字典
            
        Returns:
            模块度值
        """
        m = G.number_of_edges()
        if m == 0:
            return 0.0
            
        # 计算总度数
        total_degree = sum(dict(G.degree()).values())
        
        # 计算模块度
        Q = 0.0
        for u, v in G.edges():
            if communities[u] == communities[v]:
                # 计算边的权重（如果有）
                weight = G[u][v].get('weight', 1.0)
                # 计算节点的度数
                ku = G.degree(u)
                kv = G.degree(v)
                # 累加模块度贡献
                Q += weight - (ku * kv) / (2 * m)
                
        return Q / (2 * m)
        
    def _find_best_community(self, G: nx.Graph, node: int, communities: Dict) -> int:
        """
        为节点找到最佳社区
        
        Args:
            G: 网络图
            node: 当前节点
            communities: 社区分配字典
            
        Returns:
            最佳社区ID
        """
        # 获取节点的邻居
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return communities[node]
            
        # 计算当前社区
        current_community = communities[node]
        
        # 计算移动到每个邻居社区的变化
        best_community = current_community
        best_gain = 0.0
        
        # 获取所有可能的社区
        possible_communities = set(communities[n] for n in neighbors)
        possible_communities.add(current_community)
        
        for community in possible_communities:
            # 计算移动增益
            gain = self._compute_move_gain(G, node, community, communities)
            if gain > best_gain:
                best_gain = gain
                best_community = community
                
        return best_community
        
    def _compute_move_gain(self, G: nx.Graph, node: int, community: int, communities: Dict) -> float:
        """
        计算移动节点到新社区的增益
        
        Args:
            G: 网络图
            node: 要移动的节点
            community: 目标社区
            communities: 社区分配字典
            
        Returns:
            移动增益
        """
        m = G.number_of_edges()
        if m == 0:
            return 0.0
            
        # 计算节点的度数
        ki = G.degree(node)
        
        # 计算社区内部边的权重和
        sigma_in = sum(G[node][v].get('weight', 1.0) 
                      for v in G.neighbors(node) 
                      if communities[v] == community)
                      
        # 计算社区的总度数
        sigma_tot = sum(G.degree(v) for v in G.nodes() if communities[v] == community)
        
        # 计算增益
        gain = (sigma_in / (2 * m) - 
                (sigma_tot * ki) / (2 * m * m))
                
        return gain
        
    def _aggregate_network(self, G: nx.Graph, communities: Dict) -> Tuple[nx.Graph, Dict]:
        """
        聚合网络
        
        Args:
            G: 原始网络图
            communities: 社区分配字典
            
        Returns:
            聚合后的网络图和节点映射
        """
        # 创建新的聚合网络
        G_agg = nx.Graph()
        
        # 创建社区到新节点的映射
        community_to_node = {comm: i for i, comm in enumerate(set(communities.values()))}
        node_to_community = {i: comm for comm, i in community_to_node.items()}
        
        # 添加节点
        for i in range(len(community_to_node)):
            G_agg.add_node(i)
            
        # 添加边
        for u, v in G.edges():
            comm_u = communities[u]
            comm_v = communities[v]
            if comm_u != comm_v:
                # 获取聚合后的节点
                agg_u = community_to_node[comm_u]
                agg_v = community_to_node[comm_v]
                
                # 添加或更新边权重
                if G_agg.has_edge(agg_u, agg_v):
                    G_agg[agg_u][agg_v]['weight'] += G[u][v].get('weight', 1.0)
                else:
                    G_agg.add_edge(agg_u, agg_v, weight=G[u][v].get('weight', 1.0))
                    
        return G_agg, node_to_community
        
    def _refine_communities(self, G: nx.Graph, communities: Dict) -> Dict:
        """
        优化社区分配
        
        Args:
            G: 网络图
            communities: 初始社区分配
            
        Returns:
            优化后的社区分配
        """
        improved = True
        while improved:
            improved = False
            # 遍历所有节点
            for node in G.nodes():
                # 找到最佳社区
                best_community = self._find_best_community(G, node, communities)
                # 如果社区发生变化，更新分配
                if best_community != communities[node]:
                    communities[node] = best_community
                    improved = True
                    
        return communities
        
    def detect(self, G: nx.Graph) -> Dict:
        """
        执行社区检测
        
        Args:
            G: 输入网络图
            
        Returns:
            社区分配字典
        """
        try:
            # 初始化社区分配
            communities = {node: i for i, node in enumerate(G.nodes())}
            
            # 第一轮优化
            communities = self._refine_communities(G, communities)
            
            # 计算初始模块度
            best_modularity = self._compute_modularity(G, communities)
            best_communities = communities.copy()
            
            # 迭代优化
            while True:
                # 聚合网络
                G_agg, node_to_community = self._aggregate_network(G, communities)
                
                # 优化聚合网络
                agg_communities = self._refine_communities(G_agg, 
                    {node: i for i, node in enumerate(G_agg.nodes())})
                
                # 将聚合网络的社区分配映射回原始网络
                new_communities = {}
                for node in G.nodes():
                    comm = communities[node]
                    new_comm = agg_communities[node_to_community[comm]]
                    new_communities[node] = new_comm
                    
                # 计算新的模块度
                new_modularity = self._compute_modularity(G, new_communities)
                
                # 如果模块度没有提升，停止迭代
                if new_modularity <= best_modularity:
                    break
                    
                # 更新最佳结果
                best_modularity = new_modularity
                best_communities = new_communities
                communities = new_communities
                
            logger.info(f"Community detection completed with modularity: {best_modularity:.4f}")
            return best_communities
            
        except Exception as e:
            logger.error(f"Error in community detection: {str(e)}")
            raise
            
def best_partition(G: nx.Graph, resolution: float = 1.0, random_state: int = None) -> Dict:
    """
    使用Louvain算法检测社区
    
    Args:
        G: 输入网络图
        resolution: 分辨率参数
        random_state: 随机种子
        
    Returns:
        社区分配字典
    """
    detector = LouvainCommunityDetection(resolution=resolution, random_state=random_state)
    return detector.detect(G) 