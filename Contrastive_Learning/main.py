# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import pickle

# 修改导入，使用正确的类和函数名
from config import config, add_dataset_arg, setup_logging
from data_loader import EdgeDataLoader
from model import EdgeContrastiveModel, EdgeOrderDiscriminator
from train import Trainer
from edge_order import EdgeOrderer
from utils import setup_seed, save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Edge Order Prediction with Contrastive Learning")
    
    # 添加数据集选择参数
    parser = add_dataset_arg(parser)
    
    # 其他现有参数...
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"],
                        help="运行模式: 训练模型, 评估模型, 或预测边序")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--edge_file", type=str, default=None, help="边数据文件名")
    parser.add_argument("--model_dir", type=str, default=None, help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default=None, help="日志目录")
    parser.add_argument("--embed_model", type=str, default=None, help="嵌入模型路径")
    parser.add_argument("--discrim_model", type=str, default=None, help="判别器模型路径")
    parser.add_argument("--epochs_contrast", type=int, default=None, help="对比学习训练轮次")
    parser.add_argument("--epochs_discrim", type=int, default=None, help="判别器训练轮次")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--resplit", action="store_true", help="是否重新划分数据集")
    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    for arg, value in vars(args).items():
        if value is not None and hasattr(config, arg.upper()):
            setattr(config, arg.upper(), value)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置中的数据集
    if args.dataset and args.dataset in config.DATASETS:
        config.update_dataset(args.dataset)
        print(f"使用数据集: {config.CURRENT_DATASET}")
    
    # 更新配置
    update_config(args)
    
    # 设置日志
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # 设置随机种子
    setup_seed(config.SEED)
    
    logger.info(f"Running in {args.mode} mode")
    
    # 加载数据
    logger.info("Loading and preparing data...")
    data_loader = EdgeDataLoader(config.DATA_PATH, config)
    
    # 划分数据集，增加持久化功能
    split_file = os.path.join(os.path.dirname(config.DATA_PATH), 
                             f"{config.CURRENT_DATASET}_tr{config.TRAIN_RATIO}_split.pkl")
    
    if os.path.exists(split_file) and not args.resplit:
        # 加载已有的划分
        logger.info(f"Loading existing data split from {split_file}")
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            train_edges, train_times = split_data['train']
            val_edges, val_times = split_data['val'] 
            test_edges, test_times = split_data['test']
    else:
        # 执行新的划分
        logger.info("Creating new data split")
        (train_edges, train_times), (val_edges, val_times), (test_edges, test_times) = data_loader.split_data()
        
        # 保存划分结果
        split_data = {
            'train': (train_edges, train_times),
            'val': (val_edges, val_times),
            'test': (test_edges, test_times)
        }
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)
            logger.info(f"Saved data split to {split_file}")
    
    # 获取PyTorch Geometric数据
    pyg_data = data_loader.get_pyg_data()
    
    # 根据模式执行不同操作
    if args.mode == "train":
        # 初始化模型
        logger.info("Initializing models...")
        contrastive_model = EdgeContrastiveModel(
            num_nodes=data_loader.n_nodes,
            hidden_dim=config.GNN_HIDDEN_DIM,
            proj_dim=config.PROJ_DIM,
            num_gnn_layers=config.GNN_LAYERS,
            dropout=config.GNN_DROPOUT
        ).to(config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            embedding_dim=config.GNN_HIDDEN_DIM,
            hidden_dim=config.DISCRIMINATOR_HIDDEN
        ).to(config.DEVICE)
        
        # 创建训练器
        trainer = Trainer(
            contrastive_model=contrastive_model,
            discriminator_model=discriminator_model,
            config=config
        )
        
        # 训练对比学习模型
        logger.info("Training contrastive embedding model...")
        trainer.train_contrastive_model(
            data_loader=data_loader,
            train_edges=train_edges,
            train_times=train_times,
            val_edges=val_edges,
            val_times=val_times
        )
        
        # 训练判别器模型
        logger.info("Training discriminator model...")
        trainer.train_discriminator(
            data_loader=data_loader,
            train_edges=train_edges,
            train_times=train_times,
            val_edges=val_edges,
            val_times=val_times
        )
        
        logger.info("Training complete!")
        
    elif args.mode == "eval":
        # 加载模型
        logger.info("Loading trained models...")
        
        contrastive_model = EdgeContrastiveModel(
            num_nodes=data_loader.n_nodes,
            hidden_dim=config.GNN_HIDDEN_DIM,
            proj_dim=config.PROJ_DIM,
            num_gnn_layers=config.GNN_LAYERS,
            dropout=config.GNN_DROPOUT
        ).to(config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            embedding_dim=config.GNN_HIDDEN_DIM,
            hidden_dim=config.DISCRIMINATOR_HIDDEN
        ).to(config.DEVICE)
        
        # 加载模型权重
        embed_model_path = args.embed_model or config.CONTRASTIVE_MODEL_PATH
        discrim_model_path = args.discrim_model or config.DISCRIMINATOR_MODEL_PATH
        
        contrastive_model = load_model(contrastive_model, embed_model_path)
        discriminator_model = load_model(discriminator_model, discrim_model_path)
        
        # 创建边序处理器
        edge_orderer = EdgeOrderer(
            contrastive_model=contrastive_model,
            discriminator_model=discriminator_model,
            config=config
        )
        edge_orderer.set_data(pyg_data)
        
        # 评估边对预测准确率
        logger.info("Evaluating models on test data...")
        accuracy = edge_orderer.evaluate_accuracy(
            data_loader=data_loader,
            test_edges=test_edges,
            test_times=test_times
        )
        
        logger.info(f"测试结果:")
        logger.info(f"边对预测准确率: {accuracy:.4f}")
        
    elif args.mode == "predict":
        # 加载模型
        logger.info("Loading trained models...")
        
        contrastive_model = EdgeContrastiveModel(
            num_nodes=data_loader.n_nodes,
            hidden_dim=config.GNN_HIDDEN_DIM,
            proj_dim=config.PROJ_DIM,
            num_gnn_layers=config.GNN_LAYERS,
            dropout=config.GNN_DROPOUT
        ).to(config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            embedding_dim=config.GNN_HIDDEN_DIM,
            hidden_dim=config.DISCRIMINATOR_HIDDEN
        ).to(config.DEVICE)
        
        # 加载模型权重
        embed_model_path = args.embed_model or config.CONTRASTIVE_MODEL_PATH
        discrim_model_path = args.discrim_model or config.DISCRIMINATOR_MODEL_PATH
        
        contrastive_model = load_model(contrastive_model, embed_model_path)
        discriminator_model = load_model(discriminator_model, discrim_model_path)
        
        # 创建边序处理器
        edge_orderer = EdgeOrderer(
            contrastive_model=contrastive_model,
            discriminator_model=discriminator_model,
            config=config
        )
        edge_orderer.set_data(pyg_data)
        
        # 预测边生成顺序
        logger.info("Predicting edge generation order...")
        edges = data_loader.edges
        
        ordered_edges = edge_orderer.predict_order_by_voting(edges)
        
        # 保存预测结果
        result_path = os.path.join(config.LOG_DIR, f"{config.CURRENT_DATASET}_predicted_order.json")
        
        # 转换为可序列化的格式
        ordered_edges_list = [edge.tolist() for edge in ordered_edges]
        
        with open(result_path, 'w') as f:
            json.dump(ordered_edges_list, f)
        
        logger.info(f"Saved predicted edge order to {result_path}")

if __name__ == "__main__":
    main()