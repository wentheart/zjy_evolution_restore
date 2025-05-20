import torch
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

class EdgeOrderer:
    def __init__(self, contrastive_model, discriminator_model, config):
        """初始化边序预测器
        
        Args:
            contrastive_model: 对比学习模型
            discriminator_model: 边序判别器模型
            config: 配置对象
        """
        self.contrastive_model = contrastive_model
        self.discriminator_model = discriminator_model
        self.config = config
        self.device = config.DEVICE
        
        # 将模型移至设备
        self.contrastive_model.to(self.device)
        self.discriminator_model.to(self.device)
        
        # 设置为评估模式
        self.contrastive_model.eval()
        self.discriminator_model.eval()
        
        self.logger = logging.getLogger(__name__)
        self.edge_index = None
        self.edge_embeddings_cache = {}
    
    def set_data(self, pyg_data):
        """设置图数据
        
        Args:
            pyg_data: PyTorch Geometric数据对象
        """
        self.edge_index = pyg_data.edge_index.to(self.device)
        # 清空缓存
        self.edge_embeddings_cache = {}
    
    def evaluate_accuracy(self, data_loader, test_edges, test_times):
        """评估边对相对顺序预测的准确率
        
        Args:
            data_loader: 数据加载器
            test_edges: 测试集边
            test_times: 测试集时间
            
        Returns:
            accuracy: 准确率
        """
        self.logger.info("评估边对相对顺序预测准确率...")
        
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
                edge1_embedding = self.contrastive_model.encode_edges(self.edge_index, edge1)
                edge2_embedding = self.contrastive_model.encode_edges(self.edge_index, edge2)
                
                # 预测顺序
                logits = self.discriminator_model(edge1_embedding, edge2_embedding)
                _, pred = torch.max(logits, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label_batch.numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        self.logger.info(f"边对预测准确率: {accuracy:.4f}")
        
        return accuracy
    
    def predict_pairwise_order(self, edge1, edge2):
        """预测两条边的相对生成顺序
        
        Args:
            edge1: 第一条边 (node1, node2)
            edge2: 第二条边 (node1, node2)
            
        Returns:
            relation: 边的关系 (0表示edge1早于或同时于edge2, 1表示edge1晚于edge2)
            probabilities: 两个类别的概率分布
        """
        with torch.no_grad():
            # 获取或计算边嵌入
            edge1_tuple = tuple(edge1)
            if edge1_tuple in self.edge_embeddings_cache:
                edge1_embedding = self.edge_embeddings_cache[edge1_tuple]
            else:
                edge1_tensor = torch.tensor([edge1], dtype=torch.long, device=self.device)
                edge1_embedding = self.contrastive_model.encode_edges(self.edge_index, edge1_tensor)
                self.edge_embeddings_cache[edge1_tuple] = edge1_embedding
                
            edge2_tuple = tuple(edge2)
            if edge2_tuple in self.edge_embeddings_cache:
                edge2_embedding = self.edge_embeddings_cache[edge2_tuple]
            else:
                edge2_tensor = torch.tensor([edge2], dtype=torch.long, device=self.device)
                edge2_embedding = self.contrastive_model.encode_edges(self.edge_index, edge2_tensor)
                self.edge_embeddings_cache[edge2_tuple] = edge2_embedding
            
            # 预测顺序
            logits = self.discriminator_model(edge1_embedding, edge2_embedding)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            
            # 获取最可能的关系
            pred_class = np.argmax(probabilities)
            relation = pred_class  # 直接使用类别作为关系 (0或1)
            
        return relation, probabilities
    
    def predict_order_by_voting(self, edges):
        """通过投票机制预测边的生成顺序
        
        Args:
            edges: 边列表
            
        Returns:
            ordered_edges: 排序后的边列表
        """
        self.logger.info("通过投票机制预测边的生成顺序...")
        n_edges = len(edges)
        
        # 预先计算所有边的嵌入
        with torch.no_grad():
            self.logger.info("预计算边嵌入...")
            edges_tensor = torch.tensor(edges, dtype=torch.long, device=self.device)
            for i, edge in enumerate(tqdm(edges, desc="计算边嵌入")):
                edge_tensor = edges_tensor[i:i+1]
                embedding = self.contrastive_model.encode_edges(self.edge_index, edge_tensor)
                self.edge_embeddings_cache[tuple(edge)] = embedding
        
        # 计算每条边的得分（有多少其他边在它之后出现）
        self.logger.info("计算边的投票得分...")
        scores = np.zeros(n_edges)
        
        for i in tqdm(range(n_edges), desc="投票统计"):
            edge_i = edges[i]
            edge_i_embedding = self.edge_embeddings_cache[tuple(edge_i)]
            
            # 批量处理，提高效率
            batch_size = 128
            for j in range(0, n_edges, batch_size):
                # 跳过自己和已处理的边
                if j >= i:
                    continue
                    
                batch_edges = edges[j:min(j+batch_size, i)]
                if len(batch_edges) == 0:
                    continue
                    
                batch_embeddings = torch.cat([self.edge_embeddings_cache[tuple(e)] for e in batch_edges], dim=0)
                
                # 扩展edge_i_embedding以匹配批量大小
                expanded_embedding = edge_i_embedding.expand(len(batch_edges), -1)
                
                # 预测概率
                logits = self.discriminator_model(expanded_embedding, batch_embeddings)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                
                # 统计i比j晚生成的边数（得分越高的边生成越早）
                for k, prob in enumerate(probs):
                    idx = j + k
                    if prob[1] > 0.5:  # edge_i 晚于 edge_j
                        scores[idx] += 1
                    else:  # edge_i 早于 edge_j
                        scores[i] += 1
        
        # 根据得分对边进行排序（得分高的排前面 - 表示更早生成）
        sorted_indices = np.argsort(-scores)
        ordered_edges = [edges[i] for i in sorted_indices]
        
        return ordered_edges
