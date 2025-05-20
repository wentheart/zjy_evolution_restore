# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_layers, dropout):
        super(GNNEncoder, self).__init__()
        
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.dropout = dropout
        
    def forward(self, edge_index):
        """
        Args:
            edge_index: 图的边索引 [2, num_edges]
        
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        """
        x = self.embedding.weight
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return x

class EdgeContrastiveModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim, proj_dim, num_gnn_layers, dropout):
        super(EdgeContrastiveModel, self).__init__()
        
        # GNN编码器
        self.gnn_encoder = GNNEncoder(num_nodes, hidden_dim, num_gnn_layers, dropout)
        
        # 边表示生成器
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, edge_index, edges=None):
        """
        Args:
            edge_index: 图的边索引 [2, num_edges]
            edges: 需要编码的边列表 [(node1, node2), ...], 可选
            
        Returns:
            如果提供了edges，则返回这些边的嵌入；
            否则返回所有节点的嵌入
        """
        # 获取节点嵌入
        node_embeddings = self.gnn_encoder(edge_index)
        
        if edges is None:
            return node_embeddings
            
        # 计算边嵌入
        edge_embeddings = self._get_edge_embeddings(edges, node_embeddings)
        
        # 投影到对比学习空间
        edge_projections = self.projection(edge_embeddings)
        
        return edge_projections
    
    def _get_edge_embeddings(self, edges, node_embeddings):
        """计算边的嵌入表示
        
        Args:
            edges: 边列表 [(node1, node2), ...]
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            
        Returns:
            edge_embeddings: 边嵌入 [num_edges, hidden_dim]
        """
        edges = torch.tensor(edges, dtype=torch.long, device=node_embeddings.device)
        
        # 获取边两端节点的嵌入
        src_embeddings = node_embeddings[edges[:, 0]]
        dst_embeddings = node_embeddings[edges[:, 1]]
        
        # 拼接并编码
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        edge_embeddings = self.edge_encoder(edge_features)
        
        return edge_embeddings
        
    def encode_edges(self, edge_index, edges):
        """编码边的表示向量（不包含投影）
        
        Args:
            edge_index: 图的边索引
            edges: 需要编码的边列表 [(node1, node2), ...]
            
        Returns:
            edge_embeddings: 边嵌入 [num_edges, hidden_dim]
        """
        node_embeddings = self.gnn_encoder(edge_index)
        edge_embeddings = self._get_edge_embeddings(edges, node_embeddings)
        return edge_embeddings


class EdgeOrderDiscriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EdgeOrderDiscriminator, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 修改为2个类别，表示二分类
        )
        
    def forward(self, edge1_embedding, edge2_embedding):
        """判断edge1与edge2的时序关系
        
        Args:
            edge1_embedding: 第一条边的表示向量 [batch_size, embedding_dim]
            edge2_embedding: 第二条边的表示向量 [batch_size, embedding_dim]
            
        Returns:
            logits: 二分类的得分 [batch_size, 2]
            - idx 0: edge1早于或同时于edge2生成的得分(0)
            - idx 1: edge1晚于edge2生成的得分(1)
        """
        # 拼接两条边的嵌入
        features = torch.cat([edge1_embedding, edge2_embedding], dim=1)
        
        # 预测二分类概率
        logits = self.mlp(features)
        
        return logits