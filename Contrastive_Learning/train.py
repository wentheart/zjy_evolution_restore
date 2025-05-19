# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: 锚点嵌入 [batch_size, dim]
            positive: 正样本嵌入 [batch_size, dim]
            negatives: 负样本嵌入 [batch_size, num_negatives, dim]
            
        Returns:
            loss: 对比损失
        """
        # 计算正样本相似度
        pos_similarity = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # 计算负样本相似度
        batch_size, num_negatives, dim = negatives.shape
        anchor_expanded = anchor.unsqueeze(1).expand(-1, num_negatives, -1)  # [batch_size, num_negatives, dim]
        neg_similarity = torch.sum(anchor_expanded * negatives, dim=2) / self.temperature
        
        # InfoNCE损失
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)  # [batch_size, 1+num_negatives]
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)  # 正样本索引为0
        
        return nn.CrossEntropyLoss()(logits, labels)


class Trainer:
    def __init__(self, contrastive_model, discriminator_model, config):
        """初始化训练器
        
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
        
        # 优化器
        self.contrastive_optimizer = optim.Adam(
            self.contrastive_model.parameters(), 
            lr=config.CONTRASTIVE_LR
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator_model.parameters(), 
            lr=config.DISCRIMINATOR_LR
        )
        
        # 损失函数
        self.contrastive_loss_fn = ContrastiveLoss(temperature=config.TEMPERATURE)
        self.discriminator_loss_fn = nn.BCELoss()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def train_contrastive_model(self, data_loader, train_edges, train_times,
                              val_edges=None, val_times=None):
        """训练对比学习模型
        
        Args:
            data_loader: 数据加载器
            train_edges: 训练集边
            train_times: 训练集时间
            val_edges: 验证集边
            val_times: 验证集时间
        """
        self.logger.info("开始训练对比学习模型...")
        
        # 获取PyTorch Geometric数据
        pyg_data = data_loader.get_pyg_data()
        edge_index = pyg_data.edge_index.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.CONTRASTIVE_EPOCHS):
            start_time = time.time()
            
            # 设置为训练模式
            self.contrastive_model.train()
            
            # 生成对比学习样本
            anchors_idx, positives_idx, negatives_idxs = data_loader.generate_contrastive_pairs(
                train_edges, train_times
            )
            
            # 构建批次
            n_samples = len(anchors_idx)
            n_batches = (n_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
            
            total_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * self.config.BATCH_SIZE
                end_idx = min((i + 1) * self.config.BATCH_SIZE, n_samples)
                
                batch_anchors = anchors_idx[start_idx:end_idx]
                batch_positives = positives_idx[start_idx:end_idx]
                batch_negatives = negatives_idxs[start_idx:end_idx]
                
                # 获取边嵌入
                anchor_edges = train_edges[batch_anchors]
                positive_edges = train_edges[batch_positives]
                
                # 编码锚点和正样本
                anchor_embs = self.contrastive_model(edge_index, anchor_edges)
                positive_embs = self.contrastive_model(edge_index, positive_edges)
                
                # 编码负样本
                negative_embs_list = []
                for neg_indices in batch_negatives:
                    negative_edges = train_edges[neg_indices]
                    negative_embs = self.contrastive_model(edge_index, negative_edges)
                    negative_embs_list.append(negative_embs)
                
                # 堆叠负样本 [batch_size, num_neg, dim]
                negative_embs_stacked = torch.stack(negative_embs_list, dim=1)
                
                # 计算损失
                loss = self.contrastive_loss_fn(anchor_embs, positive_embs, negative_embs_stacked)
                
                # 反向传播
                self.contrastive_optimizer.zero_grad()
                loss.backward()
                self.contrastive_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            
            # 验证
            val_loss = 0.0
            if val_edges is not None and val_times is not None:
                val_loss = self._validate_contrastive(
                    data_loader, val_edges, val_times, edge_index
                )
                
                # 早停检查
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save(self.contrastive_model.state_dict(), self.config.CONTRASTIVE_MODEL_PATH)
                    self.logger.info(f"模型已保存到 {self.config.CONTRASTIVE_MODEL_PATH}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.PATIENCE:
                    self.logger.info(f"早停: {self.config.PATIENCE} 轮验证损失未改善")
                    break
            
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.CONTRASTIVE_EPOCHS} - "
                             f"Train Loss: {avg_loss:.4f} - "
                             f"Val Loss: {val_loss:.4f} - "
                             f"Time: {epoch_time:.2f}s")
            
        # 如果没有验证集，保存最后一轮的模型
        if val_edges is None or val_times is None:
            torch.save(self.contrastive_model.state_dict(), self.config.CONTRASTIVE_MODEL_PATH)
            
        self.logger.info("对比学习模型训练完成")
        
    def _validate_contrastive(self, data_loader, val_edges, val_times, edge_index):
        """验证对比学习模型
        
        Args:
            data_loader: 数据加载器
            val_edges: 验证集边
            val_times: 验证集时间
            edge_index: 图的边索引
            
        Returns:
            val_loss: 验证损失
        """
        self.contrastive_model.eval()
        
        # 生成验证样本
        anchors_idx, positives_idx, negatives_idxs = data_loader.generate_contrastive_pairs(
            val_edges, val_times
        )
        
        with torch.no_grad():
            # 构建批次
            n_samples = len(anchors_idx)
            n_batches = (n_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
            
            total_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * self.config.BATCH_SIZE
                end_idx = min((i + 1) * self.config.BATCH_SIZE, n_samples)
                
                batch_anchors = anchors_idx[start_idx:end_idx]
                batch_positives = positives_idx[start_idx:end_idx]
                batch_negatives = negatives_idxs[start_idx:end_idx]
                
                # 获取边嵌入
                anchor_edges = val_edges[batch_anchors]
                positive_edges = val_edges[batch_positives]
                
                # 编码锚点和正样本
                anchor_embs = self.contrastive_model(edge_index, anchor_edges)
                positive_embs = self.contrastive_model(edge_index, positive_edges)
                
                # 编码负样本
                negative_embs_list = []
                for neg_indices in batch_negatives:
                    negative_edges = val_edges[neg_indices]
                    negative_embs = self.contrastive_model(edge_index, negative_edges)
                    negative_embs_list.append(negative_embs)
                
                # 堆叠负样本
                negative_embs_stacked = torch.stack(negative_embs_list, dim=1)
                
                # 计算损失
                loss = self.contrastive_loss_fn(anchor_embs, positive_embs, negative_embs_stacked)
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
            
        return avg_loss
    
    def train_discriminator(self, data_loader, train_edges, train_times,
                          val_edges=None, val_times=None):
        """训练边序判别器
        
        Args:
            data_loader: 数据加载器
            train_edges: 训练集边
            train_times: 训练集时间
            val_edges: 验证集边
            val_times: 验证集时间
        """
        self.logger.info("开始训练边序判别器...")
        
        # 获取PyTorch Geometric数据
        pyg_data = data_loader.get_pyg_data()
        edge_index = pyg_data.edge_index.to(self.device)
        
        # 冻结对比学习模型
        for param in self.contrastive_model.parameters():
            param.requires_grad = False
            
        self.contrastive_model.eval()
        
        # 生成边对数据
        edge_pairs, labels = data_loader.generate_order_pairs(train_edges, train_times)
        
        # 创建数据集
        edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(edge_pairs, labels)
        data_loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # 如果有验证集，生成验证数据
        val_loader = None
        if val_edges is not None and val_times is not None:
            val_edge_pairs, val_labels = data_loader.generate_order_pairs(val_edges, val_times)
            val_edge_pairs = torch.tensor(val_edge_pairs, dtype=torch.long)
            val_labels = torch.tensor(val_labels, dtype=torch.float)
            val_dataset = TensorDataset(val_edge_pairs, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.DISCRIMINATOR_EPOCHS):
            start_time = time.time()
            
            # 设置为训练模式
            self.discriminator_model.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for edge_pair_batch, label_batch in data_loader:
                edge_pair_batch = edge_pair_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                # 获取边嵌入
                edge1_idx = edge_pair_batch[:, 0]
                edge2_idx = edge_pair_batch[:, 1]
                
                edge1 = train_edges[edge1_idx]
                edge2 = train_edges[edge2_idx]
                
                # 使用对比学习模型编码边（不使用投影头）
                with torch.no_grad():
                    edge1_embedding = self.contrastive_model.encode_edges(edge_index, edge1)
                    edge2_embedding = self.contrastive_model.encode_edges(edge_index, edge2)
                
                # 预测顺序
                pred = self.discriminator_model(edge1_embedding, edge2_embedding)
                
                # 计算损失
                loss = self.discriminator_loss_fn(pred, label_batch)
                
                # 反向传播
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()
                
                total_loss += loss.item()
                
                # 计算准确率
                pred_labels = (pred > 0.5).float()
                correct += (pred_labels == label_batch).sum().item()
                total += label_batch.size(0)
            
            avg_loss = total_loss / len(data_loader)
            train_acc = correct / total
            
            # 验证
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._validate_discriminator(val_loader, val_edges, edge_index)
                
                # 早停检查
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save(self.discriminator_model.state_dict(), self.config.DISCRIMINATOR_MODEL_PATH)
                    self.logger.info(f"模型已保存到 {self.config.DISCRIMINATOR_MODEL_PATH}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.PATIENCE:
                    self.logger.info(f"早停: {self.config.PATIENCE} 轮验证准确率未改善")
                    break
            
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.DISCRIMINATOR_EPOCHS} - "
                             f"Train Loss: {avg_loss:.4f} - "
                             f"Train Acc: {train_acc:.4f} - "
                             f"Val Acc: {val_acc:.4f} - "
                             f"Time: {epoch_time:.2f}s")
            
        # 如果没有验证集，保存最后一轮的模型
        if val_loader is None:
            torch.save(self.discriminator_model.state_dict(), self.config.DISCRIMINATOR_MODEL_PATH)
            
        self.logger.info("边序判别器训练完成")
        
    def _validate_discriminator(self, val_loader, val_edges, edge_index):
        """验证边序判别器
        
        Args:
            val_loader: 验证数据加载器
            val_edges: 验证集边
            edge_index: 图的边索引
            
        Returns:
            accuracy: 验证准确率
        """
        self.discriminator_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for edge_pair_batch, label_batch in val_loader:
                edge_pair_batch = edge_pair_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                # 获取边嵌入
                edge1_idx = edge_pair_batch[:, 0]
                edge2_idx = edge_pair_batch[:, 1]
                
                edge1 = val_edges[edge1_idx]
                edge2 = val_edges[edge2_idx]
                
                # 编码边
                edge1_embedding = self.contrastive_model.encode_edges(edge_index, edge1)
                edge2_embedding = self.contrastive_model.encode_edges(edge_index, edge2)
                
                # 预测顺序
                pred = self.discriminator_model(edge1_embedding, edge2_embedding)
                
                # 计算准确率
                pred_labels = (pred > 0.5).float()
                correct += (pred_labels == label_batch).sum().item()
                total += label_batch.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy