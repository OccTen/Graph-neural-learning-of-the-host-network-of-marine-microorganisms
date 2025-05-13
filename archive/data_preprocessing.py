"""
数据预处理模块，负责加载、清洗和转换数据
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Dict, List, Tuple, Union, Optional
import logging
import random
from config import DATA_PREPROCESSING, DATA_DIR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "preprocessing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPreprocessor")

class DataPreprocessor:
    """
    数据预处理器，负责加载、清洗和转换数据
    """
    
    def __init__(self, config: Dict = DATA_PREPROCESSING):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典，包含数据预处理的参数
        """
        self.config = config
        self.random_seed = config["random_seed"]
        self.set_random_seed()
        
        # 创建数据目录
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 初始化转换器
        self.scaler = None
        self.imputer = None
        self.encoder = None
        
        logger.info("数据预处理器初始化完成")
    
    def set_random_seed(self):
        """设置随机种子以确保可重复性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        logger.info(f"随机种子设置为 {self.random_seed}")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据框
        """
        try:
            # 根据文件扩展名选择加载方法
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            logger.info(f"成功加载数据，形状: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据，包括处理缺失值和异常值
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框
        """
        try:
            # 复制数据框以避免修改原始数据
            df_clean = df.copy()
            
            # 处理缺失值
            missing_strategy = self.config["missing_value_strategy"]
            if missing_strategy in ["mean", "median", "mode"]:
                imputer = SimpleImputer(strategy=missing_strategy)
                df_clean[self.config["feature_cols"]] = imputer.fit_transform(df_clean[self.config["feature_cols"]])
                self.imputer = imputer
                logger.info(f"使用 {missing_strategy} 策略处理缺失值")
            
            # 处理异常值
            outlier_strategy = self.config["outlier_strategy"]
            if outlier_strategy != "none":
                threshold = self.config["outlier_threshold"]
                for col in self.config["feature_cols"]:
                    if df_clean[col].dtype in [np.float64, np.int64]:
                        mean = df_clean[col].mean()
                        std = df_clean[col].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        
                        if outlier_strategy == "clip":
                            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                        elif outlier_strategy == "remove":
                            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
                logger.info(f"使用 {outlier_strategy} 策略处理异常值，阈值为 {threshold} 个标准差")
            
            logger.info(f"数据清洗完成，最终形状: {df_clean.shape}")
            return df_clean
        
        except Exception as e:
            logger.error(f"清洗数据时出错: {str(e)}")
            raise
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据
        
        Args:
            df: 输入数据框
            
        Returns:
            标准化后的数据框
        """
        try:
            # 复制数据框以避免修改原始数据
            df_norm = df.copy()
            
            # 选择标准化方法
            method = self.config["normalization_method"]
            if method == "minmax":
                scaler = MinMaxScaler()
            elif method == "standard":
                scaler = StandardScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            # 只对数值特征进行标准化
            df_norm[self.config["feature_cols"]] = scaler.fit_transform(df_norm[self.config["feature_cols"]])
            self.scaler = scaler
            
            logger.info(f"使用 {method} 方法标准化数据")
            return df_norm
        
        except Exception as e:
            logger.error(f"标准化数据时出错: {str(e)}")
            raise
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对分类特征进行编码
        
        Args:
            df: 输入数据框
            
        Returns:
            编码后的数据框
        """
        try:
            # 复制数据框以避免修改原始数据
            df_encoded = df.copy()
            
            # 选择编码方法
            method = self.config["encoding_method"]
            
            if method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df_encoded[self.config["categorical_cols"]])
                feature_names = []
                for i, col in enumerate(self.config["categorical_cols"]):
                    feature_names.extend([f"{col}_{cat}" for cat in encoder.categories_[i]])
                df_encoded = pd.concat([
                    df_encoded.drop(columns=self.config["categorical_cols"]),
                    pd.DataFrame(encoded_features, columns=feature_names, index=df_encoded.index)
                ], axis=1)
                self.encoder = encoder
            
            elif method == "label":
                for col in self.config["categorical_cols"]:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col])
                    self.encoder = encoder
            
            elif method == "target":
                # 目标编码需要标签，这里暂时使用标签编码代替
                for col in self.config["categorical_cols"]:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col])
                    self.encoder = encoder
                logger.warning("目标编码需要标签，暂时使用标签编码代替")
            
            else:
                raise ValueError(f"不支持的编码方法: {method}")
            
            logger.info(f"使用 {method} 方法编码分类特征")
            return df_encoded
        
        except Exception as e:
            logger.error(f"编码分类特征时出错: {str(e)}")
            raise
    
    def prepare_network_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        准备网络数据，包括节点特征和边索引
        
        Args:
            df: 输入数据框
            
        Returns:
            节点特征、边索引和节点映射字典
        """
        try:
            # 获取唯一的phage和host ID
            phage_ids = df[self.config["phage_col"]].unique()
            host_ids = df[self.config["host_col"]].unique()
            
            # 创建节点映射字典
            node_mapping = {}
            for i, phage_id in enumerate(phage_ids):
                node_mapping[phage_id] = i
            for i, host_id in enumerate(host_ids):
                node_mapping[host_id] = i + len(phage_ids)
            
            # 准备节点特征
            node_features = []
            for _, row in df.iterrows():
                phage_idx = node_mapping[row[self.config["phage_col"]]]
                host_idx = node_mapping[row[self.config["host_col"]]]
                
                # 提取特征
                features = row[self.config["feature_cols"]].values
                
                # 确保节点特征数组大小一致
                while len(node_features) <= max(phage_idx, host_idx):
                    node_features.append(np.zeros_like(features))
                
                # 更新节点特征
                node_features[phage_idx] = features
                node_features[host_idx] = features
            
            # 转换为numpy数组
            node_features = np.array(node_features)
            
            # 准备边索引
            edge_indices = []
            for _, row in df.iterrows():
                phage_idx = node_mapping[row[self.config["phage_col"]]]
                host_idx = node_mapping[row[self.config["host_col"]]]
                edge_indices.append([phage_idx, host_idx])
                edge_indices.append([host_idx, phage_idx])  # 添加反向边
            
            # 转换为numpy数组
            edge_indices = np.array(edge_indices)
            
            logger.info(f"准备网络数据完成: {len(node_features)} 个节点, {len(edge_indices)} 条边")
            return node_features, edge_indices, node_mapping
        
        except Exception as e:
            logger.error(f"准备网络数据时出错: {str(e)}")
            raise
    
    def save_preprocessed_data(self, df: pd.DataFrame, filename: str = "preprocessed_data.csv"):
        """
        保存预处理后的数据
        
        Args:
            df: 预处理后的数据框
            filename: 文件名
        """
        try:
            file_path = os.path.join(DATA_DIR, filename)
            df.to_csv(file_path, index=False)
            logger.info(f"预处理后的数据已保存到 {file_path}")
        
        except Exception as e:
            logger.error(f"保存预处理后的数据时出错: {str(e)}")
            raise
    
    def save_transformers(self):
        """保存数据转换器（标准化器、编码器等）"""
        try:
            import joblib
            
            # 保存标准化器
            if self.scaler is not None:
                scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"标准化器已保存到 {scaler_path}")
            
            # 保存缺失值填充器
            if self.imputer is not None:
                imputer_path = os.path.join(DATA_DIR, "imputer.pkl")
                joblib.dump(self.imputer, imputer_path)
                logger.info(f"缺失值填充器已保存到 {imputer_path}")
            
            # 保存编码器
            if self.encoder is not None:
                encoder_path = os.path.join(DATA_DIR, "encoder.pkl")
                joblib.dump(self.encoder, encoder_path)
                logger.info(f"编码器已保存到 {encoder_path}")
        
        except Exception as e:
            logger.error(f"保存数据转换器时出错: {str(e)}")
            raise

def main():
    """主函数，用于测试数据预处理器"""
    try:
        # 初始化数据预处理器
        preprocessor = DataPreprocessor()
        
        # 加载数据
        data_path = os.path.join(DATA_DIR, "raw_data.csv")
        df = preprocessor.load_data(data_path)
        
        # 清洗数据
        df_clean = preprocessor.clean_data(df)
        
        # 标准化数据
        df_norm = preprocessor.normalize_data(df_clean)
        
        # 编码分类特征
        df_encoded = preprocessor.encode_categorical(df_norm)
        
        # 准备网络数据
        node_features, edge_indices, node_mapping = preprocessor.prepare_network_data(df_encoded)
        
        # 保存预处理后的数据
        preprocessor.save_preprocessed_data(df_encoded)
        
        # 保存数据转换器
        preprocessor.save_transformers()
        
        logger.info("数据预处理完成")
    
    except Exception as e:
        logger.error(f"数据预处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 