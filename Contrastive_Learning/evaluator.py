# evaluator.py
import torch
import numpy as np
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, TensorDataset

class Evaluator:
    def __init__(self, contrastive_model, discriminator_model, config):
        """初始化评估器
        
        Args:
            contrastive_model: 对比学习模型
            discriminator_model: 边序判别器模型
            config: 配置对象
        """
        self.contrastive_model = contrastive_model
        self.discriminator_model = discriminator_model
        self.config = config
        self.device = config.DEVICE
        
        self.contrastive_model.to(self.device)
        self.discriminator_model.to(self.device)
        
        self.logger = logging.getLogger(__name__)
        
        # 设置为评估模式
        self.contrastive_model.eval()
        self.discriminator_model.eval()
        
    def evaluate_pairwise_accuracy(self, data_loader, test_edges, test_times):
        """评估边对相对顺序预测的准确率
        
        Args:
            data_loader: 数据加载器
            test_edges: 测试集边
            test_times: 测试集时间
            
        Returns:
            accuracy: 准确率
            f1_macro: 宏平均F1分数
            confusion: 混淆矩阵
        """
        self.logger.info("评估边对相对顺序预测准确率...")
        
        # 获取PyTorch Geometric数据
        pyg_data = data_loader.get_pyg_data()
        edge_index = pyg_data.edge_index.to(self.device)
        
        # 生成测试边对数据
        edge_pairs, labels = data_loader.generate_order_pairs(test_edges, test_times)
        
        # 创建数据集
        edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(edge_pairs, labels)
        data_loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for edge_pair_batch, label_batch in data_loader:
                edge_pair_batch = edge_pair_batch.to(self.device)
                
                # 获取边嵌入
                edge1_idx = edge_pair_batch[:, 0].cpu()
                edge2_idx = edge_pair_batch[:, 1].cpu()
                
                edge1 = test_edges[edge1_idx]
                edge2 = test_edges[edge2_idx]
                
                # 编码边
                edge1_embedding = self.contrastive_model.encode_edges(edge_index, edge1)
                edge2_embedding = self.contrastive_model.encode_edges(edge_index, edge2)
                
                # 预测顺序
                logits = self.discriminator_model(edge1_embedding, edge2_embedding)
                _, pred = torch.max(logits, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label_batch.numpy())
        
        # 计算评估指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        confusion = confusion_matrix(all_labels, all_preds)
        
        self.logger.info(f"边对预测准确率: {accuracy:.4f}")
        self.logger.info(f"边对预测宏平均F1分数: {f1_macro:.4f}")
        self.logger.info(f"混淆矩阵:\n{confusion}")
        
        # 计算每个类别的精确率和召回率
        precision = precision_score(all_labels, all_preds, average=None)
        recall = recall_score(all_labels, all_preds, average=None)
        
        self.logger.info(f"各类别精确率: 早于或同时={precision[0]:.4f}, 晚于={precision[1]:.4f}")
        self.logger.info(f"各类别召回率: 早于或同时={recall[0]:.4f}, 晚于={recall[1]:.4f}")
        
        return accuracy, f1_macro, confusion
        
    def evaluate_ranking_correlation(self, predictor, test_edges, test_times):
        """评估预测的边生成顺序与真实顺序的相关性
        
        Args:
            predictor: 顺序预测器
            test_edges: 测试集边
            test_times: 测试集时间
            
        Returns:
            kendall_tau: Kendall's Tau相关系数
            spearman: Spearman相关系数
        """
        self.logger.info("评估预测排序与真实排序的相关性...")
        
        # 使用predictor预测完整顺序
        predicted_order = predictor.predict_complete_order(test_edges)
        
        # 构建真实顺序字典
        true_order = {}
        for i, edge in enumerate(test_edges):
            true_order[tuple(edge)] = test_times[i]
        
        # 提取预测顺序和真实顺序
        true_ranks = []
        pred_ranks = []
        
        for i, edge in enumerate(predicted_order):
            true_ranks.append(true_order[tuple(edge)])
            pred_ranks.append(i)
        
        # 计算相关系数
        kendall, _ = kendalltau(true_ranks, pred_ranks)
        spearman, _ = spearmanr(true_ranks, pred_ranks)
        
        self.logger.info(f"Kendall's Tau: {kendall:.4f}")
        self.logger.info(f"Spearman相关系数: {spearman:.4f}")
        
        return kendall, spearman
        
    def visualize_predictions(self, predictor, test_edges, test_times, num_samples=50):
        """可视化部分预测结果
        
        Args:
            predictor: 顺序预测器
            test_edges: 测试集边
            test_times: 测试集时间
            num_samples: 样本数量
        """
        self.logger.info("可视化部分预测结果...")
        
        # 随机采样一些边对
        n_edges = len(test_edges)
        indices = np.random.choice(n_edges, min(num_samples, n_edges), replace=False)
        
        sample_edges = test_edges[indices]
        sample_times = test_times[indices]
        
        # 预测这些边对之间的相对顺序
        results = []
        
        for i, edge1 in enumerate(sample_edges):
            edge1_tuple = tuple(edge1)
            for j, edge2 in enumerate(sample_edges):
                if i == j:
                    continue
                    
                edge2_tuple = tuple(edge2)
                
                # 计算真实关系（二分类）
                time_diff = sample_times[i] - sample_times[j]

                if abs(time_diff) > predictor.config.TIME_THRESHOLD:
                    if time_diff < -predictor.config.TIME_THRESHOLD:
                        true_relation = 1  # edge1晚于edge2
                    else:
                        true_relation = 0  # edge1早于或同时于edge2
                    
                # 预测edge1与edge2的关系
                pred_relation, probabilities = predictor.predict_pairwise_order(edge1, edge2)
                
                results.append({
                    'edge1': edge1_tuple,
                    'edge2': edge2_tuple,
                    'true_time1': sample_times[i],
                    'true_time2': sample_times[j],
                    'true_relation': true_relation,
                    'pred_relation': pred_relation,
                    'probabilities': probabilities,
                    'correct': (true_relation == pred_relation)
                })
        
        # 打印结果
        self.logger.info("边对预测结果示例:")
        for i, result in enumerate(results[:50]):  # 只显示前10个
            self.logger.info(f"样本 {i+1}:")
            self.logger.info(f"  边1: {result['edge1']}, 时间: {result['true_time1']}")
            self.logger.info(f"  边2: {result['edge2']}, 时间: {result['true_time2']}")
            
            # 显示真实关系
            if result['true_relation'] == 1:
                true_rel_text = "边1晚于边2"
            else:
                true_rel_text = "边1早于或同时于边2"
                
            # 显示预测关系
            if result['pred_relation'] == 1:
                pred_rel_text = "边1晚于边2"
            else:
                pred_rel_text = "边1早于或同时于边2"
                
            self.logger.info(f"  真实关系: {true_rel_text}")
            self.logger.info(f"  预测关系: {pred_rel_text}")
            self.logger.info(f"  预测概率: 早于或同时={result['probabilities'][0]:.4f}, "
                            f"晚于={result['probabilities'][1]:.4f}")
            self.logger.info(f"  是否正确: {'✓' if result['correct'] else '✗'}")
            self.logger.info("")