# utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import logging
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import json
from config import Config

def setup_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def setup_logger(log_dir, name="edge_order_prediction"):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"{name}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def save_model(model, path):
    """保存模型到指定路径"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    """从指定路径加载模型"""
    model.load_state_dict(torch.load(path))
    return model

def save_training_history(history, path):
    """保存训练历史到JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f)

def plot_training_history(history, save_path=None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制准确率曲线
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()

def visualize_graph(edges, edge_times=None, node_size=300, figsize=(12, 10), save_path=None):
    """可视化图结构，可选显示边的生成时间"""
    G = nx.Graph()
    
    # 添加边到图
    for i, (u, v) in enumerate(edges):
        G.add_edge(u, v)
        
    # 绘制图
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')
    
    # 如果提供了边时间，按时间上色
    if edge_times:
        # 归一化时间值用于颜色映射
        norm_times = [(t - min(edge_times)) / (max(edge_times) - min(edge_times) + 1e-10) for t in edge_times]
        
        # 绘制边，颜色表示生成时间
        for i, (u, v) in enumerate(edges):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                  width=2, alpha=0.7, 
                                  edge_color=[norm_times[i]], 
                                  edge_cmap=plt.cm.viridis)
    else:
        # 无时间信息时统一颜色
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Graph Structure" + (" with Edge Generation Times" if edge_times else ""))
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()

def compute_kendall_tau(pred_order, true_order):
    """
    计算Kendall's Tau相关系数，评估两个序列的排序相似度
    
    Args:
        pred_order: 预测的边顺序（索引列表）
        true_order: 真实的边顺序（索引列表）
        
    Returns:
        tau: Kendall's Tau相关系数 (-1到1，1表示完全一致)
    """
    # 创建排名映射
    pred_ranks = {edge_idx: rank for rank, edge_idx in enumerate(pred_order)}
    true_ranks = {edge_idx: rank for rank, edge_idx in enumerate(true_order)}
    
    # 确保两个排名包含相同的项
    common_items = set(pred_ranks.keys()) & set(true_ranks.keys())
    
    # 提取排名
    pred_rank_list = [pred_ranks[item] for item in common_items]
    true_rank_list = [true_ranks[item] for item in common_items]
    
    # 计算Kendall's Tau
    tau, p_value = stats.kendalltau(pred_rank_list, true_rank_list)
    
    return tau

def evaluate_pairwise_accuracy(pred_pairs, true_pairs):
    """
    评估边对相对顺序的预测准确率
    
    Args:
        pred_pairs: 预测的边对相对顺序 [(边1索引, 边2索引, 预测标签), ...]
        true_pairs: 真实的边对相对顺序 [(边1索引, 边2索引, 真实标签), ...]
        
    Returns:
        accuracy: 预测准确率
    """
    # 创建边对到预测标签的映射
    pred_dict = {(e1, e2): label for e1, e2, label in pred_pairs}
    
    # 计算准确的预测数
    correct = 0
    total = 0
    
    for e1, e2, true_label in true_pairs:
        if (e1, e2) in pred_dict:
            pred_label = pred_dict[(e1, e2)]
            if pred_label == true_label:
                correct += 1
            total += 1
        elif (e2, e1) in pred_dict:
            # 如果顺序相反，标签也要相反
            pred_label = 1 - pred_dict[(e2, e1)]
            if pred_label == true_label:
                correct += 1
            total += 1
    
    if total == 0:
        return 0.0
        
    return correct / total