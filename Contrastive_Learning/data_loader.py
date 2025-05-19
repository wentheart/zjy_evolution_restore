# data_loader.py
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split

class EdgeDataLoader:
    def __init__(self, file_path, config):
        """初始化数据加载器
        
        Args:
            file_path: 数据文件路径
            config: 配置对象
        """
        self.file_path = file_path
        self.config = config
        self.edges = []              # [(node1, node2), ...]
        self.edge_times = []         # [time1, time2, ...]
        self.n_nodes = 0
        self.time_threshold = config.TIME_THRESHOLD
        
        # 加载数据
        self._load_data()
        
        # 构建图
        self._build_graph()
        
    def _load_data(self):
        """加载边数据"""
        with open(self.file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                # 解析边和时间信息
                edge_str, time_str = line.strip().split(':')
                time = int(time_str)
                
                # 解析边的节点
                node_str = edge_str.strip('()')
                node1, node2 = map(int, node_str.split(','))
                
                self.edges.append((node1, node2))
                self.edge_times.append(time)
                
                # 更新节点数
                self.n_nodes = max(self.n_nodes, node1 + 1, node2 + 1)
        
        # 转换为numpy数组
        self.edges = np.array(self.edges)
        self.edge_times = np.array(self.edge_times)
        
    def _build_graph(self):
        """构建图结构"""
        # 使用NetworkX构建图
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_nodes))
        
        for (src, dst), time in zip(self.edges, self.edge_times):
            self.G.add_edge(src, dst, time=time)
        
        # 创建PyTorch Geometric数据对象
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(self.edge_times, dtype=torch.float).unsqueeze(1)
        
        self.data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_nodes
        )
    
    def split_data(self):
        """划分训练、验证和测试集"""
        n_edges = len(self.edges)
        indices = list(range(n_edges))
        
        # 首先划分训练集和临时集
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=self.config.TRAIN_RATIO,
            random_state=self.config.SEED
        )
        
        # 然后从临时集中划分验证集和测试集
        val_size = self.config.VAL_RATIO / (1 - self.config.TRAIN_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=self.config.SEED
        )
        
        # 分割数据
        train_edges = self.edges[train_idx]
        train_times = self.edge_times[train_idx]
        
        val_edges = self.edges[val_idx]
        val_times = self.edge_times[val_idx]
        
        test_edges = self.edges[test_idx]
        test_times = self.edge_times[test_idx]
        
        return (train_edges, train_times), (val_edges, val_times), (test_edges, test_times)
    
    def generate_contrastive_pairs(self, edges, times):
        """为对比学习生成正负样本对
        
        Args:
            edges: 边列表 [(node1, node2), ...]
            times: 对应的时间列表 [time1, time2, ...]
            
        Returns:
            anchors: 锚点边索引列表
            positives: 正样本边索引列表
            negatives: 负样本边索引列表（每个锚点对应多个）
        """
        n_edges = len(edges)
        anchors = []
        positives = []
        negatives_list = []
        
        time_array = np.array(times)
        
        # 对每条边生成正负样本
        for i in range(n_edges):
            anchor_time = time_array[i]
            
            # 寻找正样本：时间差小于等于阈值的边
            pos_candidates = np.where(np.abs(time_array - anchor_time) <= self.time_threshold)[0]
            # 排除自身
            pos_candidates = pos_candidates[pos_candidates != i]
            
            if len(pos_candidates) == 0:
                continue
                
            # 寻找负样本：时间差大于阈值的边
            neg_candidates = np.where(np.abs(time_array - anchor_time) > self.time_threshold)[0]
            
            if len(neg_candidates) < self.config.NUM_NEG_SAMPLES:
                continue
            
            # 采样一个正样本
            pos_idx = np.random.choice(pos_candidates)
            
            # 采样多个负样本
            neg_indices = np.random.choice(
                neg_candidates, 
                size=self.config.NUM_NEG_SAMPLES, 
                replace=False
            )
            
            anchors.append(i)
            positives.append(pos_idx)
            negatives_list.append(neg_indices)
            
        return anchors, positives, negatives_list
        
    def generate_order_pairs(self, edges, times):
        """生成用于训练判别器的边对数据
        
        Args:
            edges: 边列表
            times: 时间列表
            
        Returns:
            edge_pairs: 边对索引 [(idx1, idx2), ...]
            labels: 标签 [1 if edges[idx1]早于edges[idx2]生成 else 0, ...]
        """
        n_edges = len(edges)
        edge_pairs = []
        labels = []
        
        # 生成一定数量的边对
        num_pairs = min(10000, n_edges * 5)  # 限制对的数量
        
        for _ in range(num_pairs):
            # 随机选择两条不同的边
            i, j = random.sample(range(n_edges), 2)
            
            edge_pairs.append((i, j))
            
            # 如果edge_i的时间早于edge_j，标签为1，否则为0
            if times[i] < times[j]:
                labels.append(1)
            else:
                labels.append(0)
        
        return edge_pairs, labels
        
    def get_pyg_data(self):
        """获取PyTorch Geometric数据对象"""
        return self.data
        
    def get_edge_time_dict(self):
        """返回边到时间的映射字典"""
        edge_time_dict = {}
        for (src, dst), time in zip(self.edges, self.edge_times):
            edge_time_dict[(src, dst)] = time
            edge_time_dict[(dst, src)] = time  # 无向图
        return edge_time_dict