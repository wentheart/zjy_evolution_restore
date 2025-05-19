# predictor.py
import torch
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm

class OrderPredictor:
    def __init__(self, contrastive_model, discriminator_model, config):
        """初始化顺序预测器
        
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
        self.pyg_data = None
        self.edge_embeddings_cache = {}
        
    def set_data(self, pyg_data):
        """设置图数据
        
        Args:
            pyg_data: PyTorch Geometric数据对象
        """
        self.pyg_data = pyg_data
        self.edge_index = pyg_data.edge_index.to(self.device)
        # 清空缓存
        self.edge_embeddings_cache = {}
        
    def predict_pairwise_order(self, edge1, edge2):
        """预测两条边的相对生成顺序
        
        Args:
            edge1: 第一条边 (node1, node2)
            edge2: 第二条边 (node1, node2)
            
        Returns:
            probability: edge1早于edge2生成的概率
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
            probability = self.discriminator_model(edge1_embedding, edge2_embedding).item()
            
        return probability
        
    def predict_all_pairwise_orders(self, edges):
        """预测所有边对之间的相对顺序
        
        Args:
            edges: 边列表
            
        Returns:
            order_matrix: 顺序矩阵，order_matrix[i, j] = 1表示边i早于边j
        """
        n_edges = len(edges)
        self.logger.info(f"预测 {n_edges} 条边的所有相对顺序...")
        
        # 初始化顺序矩阵
        order_matrix = np.zeros((n_edges, n_edges))
        
        # 预先计算所有边的嵌入
        with torch.no_grad():
            edges_tensor = torch.tensor(edges, dtype=torch.long, device=self.device)
            for i, edge in enumerate(edges):
                edge_tensor = edges_tensor[i:i+1]
                embedding = self.contrastive_model.encode_edges(self.edge_index, edge_tensor)
                self.edge_embeddings_cache[tuple(edge)] = embedding
        
        # 计算所有边对的顺序
        for i in tqdm(range(n_edges), desc="预测边对顺序"):
            edge_i = edges[i]
            edge_i_embedding = self.edge_embeddings_cache[tuple(edge_i)]
            
            # 批量处理，提高效率
            batch_size = 128
            for j in range(0, n_edges, batch_size):
                batch_edges = edges[j:min(j+batch_size, n_edges)]
                batch_embeddings = torch.cat([self.edge_embeddings_cache[tuple(e)] for e in batch_edges], dim=0)
                
                # 扩展edge_i_embedding以匹配批量大小
                expanded_embedding = edge_i_embedding.expand(len(batch_edges), -1)
                
                # 预测概率
                probabilities = self.discriminator_model(expanded_embedding, batch_embeddings).cpu().numpy()
                
                # 填充顺序矩阵
                for k, prob in enumerate(probabilities):
                    idx = j + k
                    if i != idx:  # 跳过自身
                        # edge_i早于edge_j的概率大于0.5，则认为edge_i早于edge_j
                        order_matrix[i, idx] = 1 if prob > 0.5 else 0
        
        return order_matrix
        
    def build_precedence_graph(self, order_matrix):
        """根据顺序矩阵构建优先级图
        
        Args:
            order_matrix: 顺序矩阵
            
        Returns:
            G: NetworkX有向图
        """
        n = order_matrix.shape[0]
        G = nx.DiGraph()
        
        # 添加节点
        G.add_nodes_from(range(n))
        
        # 添加有向边
        for i in range(n):
            for j in range(n):
                if i != j and order_matrix[i, j] == 1:
                    G.add_edge(i, j)
        
        return G
        
    def resolve_contradictions(self, order_matrix):
        """解决顺序矩阵中的矛盾
        
        Args:
            order_matrix: 顺序矩阵
            
        Returns:
            resolved_matrix: 解决矛盾后的顺序矩阵
        """
        n = order_matrix.shape[0]
        resolved_matrix = order_matrix.copy()
        
        # 检查并解决矛盾
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 如果i早于j且j早于i，则矛盾
                    if resolved_matrix[i, j] == 1 and resolved_matrix[j, i] == 1:
                        # 根据概率值解决矛盾
                        prob_i_j = self.predict_pairwise_order(i, j)
                        prob_j_i = 1 - prob_i_j
                        
                        # 保留概率较高的顺序
                        if prob_i_j >= prob_j_i:
                            resolved_matrix[j, i] = 0
                        else:
                            resolved_matrix[i, j] = 0
        
        return resolved_matrix
        
    def topological_sort(self, order_matrix):
        """根据顺序矩阵进行拓扑排序
        
        Args:
            order_matrix: 顺序矩阵
            
        Returns:
            sorted_indices: 排序后的边索引
        """
        # 构建优先级图
        G = self.build_precedence_graph(order_matrix)
        
        # 检查是否有环
        if nx.is_directed_acyclic_graph(G):
            # 拓扑排序
            return list(nx.topological_sort(G))
        else:
            # 如果有环，解决矛盾
            self.logger.info("优先级图中存在环，尝试解决矛盾...")
            resolved_matrix = self.resolve_contradictions(order_matrix)
            G = self.build_precedence_graph(resolved_matrix)
            
            # 再次检查是否有环
            if nx.is_directed_acyclic_graph(G):
                return list(nx.topological_sort(G))
            else:
                self.logger.warning("无法解决所有矛盾，使用Kahn算法进行近似拓扑排序")
                # 使用Kahn算法进行近似拓扑排序
                return self._approximate_topological_sort(G)
    
    def _approximate_topological_sort(self, G):
        """当图中存在环时，进行近似拓扑排序
        
        Args:
            G: NetworkX图
            
        Returns:
            sorted_nodes: 排序后的节点列表
        """
        # 复制图，因为算法会修改图
        H = G.copy()
        
        result = []
        # 找出所有入度为0的节点
        sources = [n for n in H if H.in_degree(n) == 0]
        
        while sources:
            # 从图中移除一个入度为0的节点
            n = sources.pop(0)
            result.append(n)
            
            # 对于这个节点的所有邻居
            for m in list(H.successors(n)):
                # 移除边
                H.remove_edge(n, m)
                
                # 如果m的入度变为0，添加到sources
                if H.in_degree(m) == 0:
                    sources.append(m)
        
        # 检查是否有剩余边（有环）
        if H.number_of_edges() > 0:
            # 使用PageRank打破环
            pr = nx.pagerank(H)
            remaining_nodes = sorted(H.nodes(), key=lambda x: -pr[x])
            result.extend(remaining_nodes)
            
            # 移除重复
            result = list(dict.fromkeys(result))
        
        return result
        
    def predict_complete_order(self, edges):
        """预测边的完整生成顺序
        
        Args:
            edges: 边列表
            
        Returns:
            ordered_edges: 排序后的边列表
        """
        self.logger.info("预测完整边生成顺序...")
        
        # 预测所有边对的顺序
        order_matrix = self.predict_all_pairwise_orders(edges)
        
        # 拓扑排序
        sorted_indices = self.topological_sort(order_matrix)
        
        # 获取排序后的边
        ordered_edges = [edges[i] for i in sorted_indices]
        
        return ordered_edges