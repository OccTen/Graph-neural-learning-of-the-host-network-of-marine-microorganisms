"""
评估模块，用于模型评估和结果可视化
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import torch
from torch_geometric.data import Data
from config import EVALUATION_CONFIG, RESULTS_DIR, DATA_DIR, MODEL_DIR, LOGGING_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    filename=LOGGING_CONFIG['file']
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估类，用于评估模型性能并可视化结果"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or EVALUATION_CONFIG
        self.metrics = {}
        self.predictions = None
        self.true_labels = None
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        for directory in [RESULTS_DIR, DATA_DIR, MODEL_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"确保目录存在: {directory}")
            
    def evaluate(self, model: torch.nn.Module, data: Data, device: str) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 模型
            data: 评估数据
            device: 设备（'cuda' 或 'cpu'）
            
        Returns:
            评估指标字典
        """
        try:
            # 将数据移到指定设备
            data = data.to(device)
            
            # 设置为评估模式
            model.eval()
            
            with torch.no_grad():
                # 获取预测结果
                out = model(data.x, data.edge_index)
                pred = (out.squeeze() > self.config['threshold']).float()
                
                # 保存预测结果和真实标签
                self.predictions = pred.cpu().numpy()
                self.true_labels = data.edge_attr.cpu().numpy()
                
                # 计算评估指标
                metrics = {}
                for metric in self.config['metrics']:
                    if metric == 'accuracy':
                        metrics[metric] = accuracy_score(self.true_labels, self.predictions)
                    elif metric == 'precision':
                        metrics[metric] = precision_score(self.true_labels, self.predictions)
                    elif metric == 'recall':
                        metrics[metric] = recall_score(self.true_labels, self.predictions)
                    elif metric == 'f1':
                        metrics[metric] = f1_score(self.true_labels, self.predictions)
                    elif metric == 'auc':
                        metrics[metric] = roc_auc_score(self.true_labels, out.squeeze().cpu().numpy())
                
                self.metrics = metrics
                logger.info("模型评估完成")
                logger.info(f"评估指标: {metrics}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"评估过程中出错: {str(e)}")
            raise
            
    def plot_training_curves(self, history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        """
        绘制训练曲线
        
        Args:
            history: 训练历史记录
            save_path: 保存路径（可选）
        """
        if not self.config['plot_curves']:
            logger.info("配置中禁用了曲线绘制")
            return
            
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='训练损失')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='验证损失')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.title('训练和验证损失曲线')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"训练曲线已保存到: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {str(e)}")
            raise
            
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径（可选）
        """
        if not self.config['plot_curves']:
            logger.info("配置中禁用了混淆矩阵绘制")
            return
            
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('混淆矩阵')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"混淆矩阵已保存到: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制混淆矩阵时出错: {str(e)}")
            raise
            
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_pred: 预测概率
            save_path: 保存路径（可选）
        """
        if not self.config['plot_curves']:
            logger.info("配置中禁用了ROC曲线绘制")
            return
            
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('ROC曲线')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"ROC曲线已保存到: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制ROC曲线时出错: {str(e)}")
            raise
            
    def save_predictions(self, predictions: np.ndarray, save_path: str) -> None:
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            save_path: 保存路径
        """
        if not self.config['save_predictions']:
            logger.info("配置中禁用了预测结果保存")
            return
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存预测结果
            df = pd.DataFrame({
                'prediction': predictions
            })
            df.to_csv(save_path, index=False)
            logger.info(f"预测结果已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存预测结果时出错: {str(e)}")
            raise
            
    def save_metrics(self, metrics: Dict[str, float], save_path: str) -> None:
        """
        保存评估指标
        
        Args:
            metrics: 评估指标字典
            save_path: 保存路径
        """
        if not self.config['save_metrics']:
            logger.info("配置中禁用了评估指标保存")
            return
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存评估指标
            df = pd.DataFrame([metrics])
            df.to_csv(save_path, index=False)
            logger.info(f"评估指标已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存评估指标时出错: {str(e)}")
            raise
            
    def visualize_results(self, save_dir: Optional[str] = None) -> None:
        """
        可视化所有结果
        
        Args:
            save_dir: 保存目录（可选）
        """
        try:
            if save_dir is None:
                save_dir = RESULTS_DIR
                
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 绘制训练曲线
            if hasattr(self, 'train_losses') and hasattr(self, 'val_losses'):
                self.plot_training_curves(
                    {'train_loss': self.train_losses, 'val_loss': self.val_losses},
                    os.path.join(save_dir, 'training_curves.png')
                )
            
            # 绘制混淆矩阵
            if self.predictions is not None and self.true_labels is not None:
                self.plot_confusion_matrix(
                    self.true_labels,
                    self.predictions,
                    os.path.join(save_dir, 'confusion_matrix.png')
                )
            
            # 绘制ROC曲线
            if self.predictions is not None and self.true_labels is not None:
                self.plot_roc_curve(
                    self.true_labels,
                    self.predictions,
                    os.path.join(save_dir, 'roc_curve.png')
                )
            
            # 保存预测结果
            if self.predictions is not None and self.config['save_predictions']:
                self.save_predictions(
                    self.predictions,
                    os.path.join(save_dir, 'predictions.csv')
                )
            
            # 保存评估指标
            if self.metrics and self.config['save_metrics']:
                self.save_metrics(
                    self.metrics,
                    os.path.join(save_dir, 'metrics.csv')
                )
            
            logger.info(f"所有结果已可视化并保存到: {save_dir}")
            
        except Exception as e:
            logger.error(f"可视化结果时出错: {str(e)}")
            raise

def main():
    """主函数，用于测试评估器"""
    try:
        # 加载数据
        data = torch.load(os.path.join(DATA_DIR, "processed_graph.pt"))
        
        # 加载模型
        model = torch.load(os.path.join(MODEL_DIR, "best_model.pt"))
        
        # 创建评估器
        evaluator = ModelEvaluator()
        
        # 评估模型
        metrics = evaluator.evaluate(model, data, device='cuda')
        
        # 打印评估指标
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 可视化结果
        evaluator.visualize_results()
        
        logger.info("模型评估完成")
        
    except Exception as e:
        logger.error(f"模型评估过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 