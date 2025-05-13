"""
测试数据预处理模块
"""

import os
import pandas as pd
import numpy as np
import logging
from data_preprocessing import DataPreprocessor
from config import DATA_DIR, DATA_PREPROCESSING

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "test_preprocessing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestPreprocessing")

def create_sample_data():
    """创建示例数据用于测试"""
    # 创建示例数据
    np.random.seed(42)
    
    # 创建phage和host ID
    num_phages = 10
    num_hosts = 5
    phage_ids = [f"phage_{i}" for i in range(num_phages)]
    host_ids = [f"host_{i}" for i in range(num_hosts)]
    
    # 创建特征
    feature_cols = DATA_PREPROCESSING["feature_cols"]
    categorical_cols = DATA_PREPROCESSING["categorical_cols"]
    
    # 创建数据框
    data = []
    for phage_id in phage_ids:
        for host_id in host_ids:
            # 随机决定是否创建关系
            if np.random.random() > 0.7:  # 30%的概率创建关系
                # 创建数值特征
                features = {col: np.random.normal(0, 1) for col in feature_cols}
                
                # 创建分类特征
                for col in categorical_cols:
                    if col == "family":
                        features[col] = np.random.choice(["A", "B", "C"])
                    elif col == "order":
                        features[col] = np.random.choice(["X", "Y", "Z"])
                    elif col == "genus":
                        features[col] = np.random.choice(["alpha", "beta", "gamma"])
                
                # 添加ID
                features["phage_id"] = phage_id
                features["host_id"] = host_id
                
                data.append(features)
    
    # 转换为数据框
    df = pd.DataFrame(data)
    
    # 添加一些缺失值和异常值
    # 缺失值
    for col in feature_cols:
        mask = np.random.random(df.shape[0]) < 0.1  # 10%的缺失值
        df.loc[mask, col] = np.nan
    
    # 异常值
    for col in feature_cols:
        mask = np.random.random(df.shape[0]) < 0.05  # 5%的异常值
        df.loc[mask, col] = df[col].mean() + 10 * df[col].std()
    
    return df

def test_data_preprocessing():
    """测试数据预处理模块"""
    try:
        # 创建示例数据
        logger.info("创建示例数据...")
        df = create_sample_data()
        
        # 保存原始数据
        raw_data_path = os.path.join(DATA_DIR, "raw_data.csv")
        df.to_csv(raw_data_path, index=False)
        logger.info(f"原始数据已保存到 {raw_data_path}")
        
        # 初始化数据预处理器
        logger.info("初始化数据预处理器...")
        preprocessor = DataPreprocessor()
        
        # 加载数据
        logger.info("加载数据...")
        df_loaded = preprocessor.load_data(raw_data_path)
        assert df_loaded.shape == df.shape, "加载的数据形状不正确"
        logger.info(f"成功加载数据，形状: {df_loaded.shape}")
        
        # 清洗数据
        logger.info("清洗数据...")
        df_clean = preprocessor.clean_data(df_loaded)
        assert not df_clean.isnull().any().any(), "清洗后的数据仍包含缺失值"
        logger.info(f"数据清洗完成，最终形状: {df_clean.shape}")
        
        # 标准化数据
        logger.info("标准化数据...")
        df_norm = preprocessor.normalize_data(df_clean)
        for col in DATA_PREPROCESSING["feature_cols"]:
            assert df_norm[col].min() >= 0 and df_norm[col].max() <= 1, f"{col} 列未正确标准化"
        logger.info("数据标准化完成")
        
        # 编码分类特征
        logger.info("编码分类特征...")
        df_encoded = preprocessor.encode_categorical(df_norm)
        logger.info(f"分类特征编码完成，最终形状: {df_encoded.shape}")
        
        # 准备网络数据
        logger.info("准备网络数据...")
        node_features, edge_indices, node_mapping = preprocessor.prepare_network_data(df_encoded)
        logger.info(f"网络数据准备完成: {len(node_features)} 个节点, {len(edge_indices)} 条边")
        
        # 保存预处理后的数据
        logger.info("保存预处理后的数据...")
        preprocessor.save_preprocessed_data(df_encoded)
        
        # 保存数据转换器
        logger.info("保存数据转换器...")
        preprocessor.save_transformers()
        
        logger.info("数据预处理测试完成")
        return True
    
    except Exception as e:
        logger.error(f"数据预处理测试过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_data_preprocessing()
    if success:
        print("数据预处理测试成功完成！")
    else:
        print("数据预处理测试失败，请查看日志文件了解详情。") 