# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json

from config import Config
from data_loader import load_and_prepare_data
from model import EdgeEmbeddingModel, EdgeOrderDiscriminator
from train import train_contrastive_model, train_discriminator
from evaluator import evaluate_model
from predictor import predict_edge_order
from utils import setup_seed, setup_logger, save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Edge Generation Order Prediction")
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
    return parser.parse_args()

def update_config(config, args):
    """根据命令行参数更新配置"""
    for arg, value in vars(args).items():
        if value is not None and hasattr(config, arg.upper()):
            setattr(config, arg.upper(), value)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    update_config(Config, args)
    
    # 创建必要的目录
    Config.create_directories()
    
    # 设置随机种子和日志记录
    setup_seed(Config.SEED)
    logger = setup_logger(Config.LOG_DIR)
    logger.info(f"Running in {args.mode} mode")
    
    # 加载和准备数据
    logger.info("Loading and preparing data...")
    data_dict = load_and_prepare_data(Config)
    
    # 根据模式执行不同操作
    if args.mode == "train":
        # 初始化模型
        logger.info("Initializing models...")
        embedding_model = EdgeEmbeddingModel(
            num_nodes=data_dict['num_nodes'],
            hidden_dim=Config.HIDDEN_DIM,
            gnn_layers=Config.GNN_LAYERS,
            projection_dim=Config.PROJECTION_DIM,
            dropout=Config.DROPOUT
        ).to(Config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            input_dim=Config.HIDDEN_DIM,
            hidden_dim=Config.DISCRIMINATOR_HIDDEN,
            dropout=Config.DROPOUT
        ).to(Config.DEVICE)
        
        # 训练对比学习模型
        logger.info("Training contrastive embedding model...")
        train_contrastive_model(
            embedding_model=embedding_model,
            train_loader=data_dict['train_contrast_loader'],
            val_loader=data_dict.get('val_contrast_loader'),
            adj_list=data_dict['adj_list'],
            config=Config,
            logger=logger
        )
        
        # 保存嵌入模型
        embed_model_path = os.path.join(Config.MODEL_DIR, "embedding_model.pt")
        save_model(embedding_model, embed_model_path)
        logger.info(f"Saved embedding model to {embed_model_path}")
        
        # 训练判别器模型
        logger.info("Training discriminator model...")
        train_discriminator(
            embedding_model=embedding_model,
            discriminator_model=discriminator_model,
            train_loader=data_dict['train_order_loader'],
            val_loader=data_dict['val_order_loader'],
            adj_list=data_dict['adj_list'],
            config=Config,
            logger=logger
        )
        
        # 保存判别器模型
        discrim_model_path = os.path.join(Config.MODEL_DIR, "discriminator_model.pt")
        save_model(discriminator_model, discrim_model_path)
        logger.info(f"Saved discriminator model to {discrim_model_path}")
        
    elif args.mode == "eval":
        # 加载模型
        logger.info("Loading trained models...")
        
        embedding_model = EdgeEmbeddingModel(
            num_nodes=data_dict['num_nodes'],
            hidden_dim=Config.HIDDEN_DIM,
            gnn_layers=Config.GNN_LAYERS,
            projection_dim=Config.PROJECTION_DIM
        ).to(Config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            input_dim=Config.HIDDEN_DIM,
            hidden_dim=Config.DISCRIMINATOR_HIDDEN
        ).to(Config.DEVICE)
        
        embed_model_path = args.embed_model or os.path.join(Config.MODEL_DIR, "embedding_model.pt")
        discrim_model_path = args.discrim_model or os.path.join(Config.MODEL_DIR, "discriminator_model.pt")
        
        embedding_model = load_model(embedding_model, embed_model_path)
        discriminator_model = load_model(discriminator_model, discrim_model_path)
        
        # 评估模型
        logger.info("Evaluating models on test data...")
        metrics = evaluate_model(
            embedding_model=embedding_model,
            discriminator_model=discriminator_model,
            test_loader=data_dict['test_order_loader'],
            adj_list=data_dict['adj_list'],
            config=Config,
            logger=logger
        )
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Kendall's Tau: {metrics['kendall_tau']:.4f}")
        
    elif args.mode == "predict":
        # 加载模型
        logger.info("Loading trained models...")
        
        embedding_model = EdgeEmbeddingModel(
            num_nodes=data_dict['num_nodes'],
            hidden_dim=Config.HIDDEN_DIM,
            gnn_layers=Config.GNN_LAYERS,
            projection_dim=Config.PROJECTION_DIM
        ).to(Config.DEVICE)
        
        discriminator_model = EdgeOrderDiscriminator(
            input_dim=Config.HIDDEN_DIM,
            hidden_dim=Config.DISCRIMINATOR_HIDDEN
        ).to(Config.DEVICE)
        
        embed_model_path = args.embed_model or os.path.join(Config.MODEL_DIR, "embedding_model.pt")
        discrim_model_path = args.discrim_model or os.path.join(Config.MODEL_DIR, "discriminator_model.pt")
        
        embedding_model = load_model(embedding_model, embed_model_path)
        discriminator_model = load_model(discriminator_model, discrim_model_path)
        
        # 预测边生成顺序
        logger.info("Predicting edge generation order...")
        edges = data_dict['dataset'].edges
        edge_times = data_dict['dataset'].edge_times
        
        predicted_order = predict_edge_order(
            edges=edges,
            embedding_model=embedding_model,
            discriminator_model=discriminator_model,
            adj_list=data_dict['adj_list'],
            config=Config,
            logger=logger
        )
        
        # 保存预测结果
        result_path = os.path.join(Config.LOG_DIR, "predicted_order.json")
        with open(result_path, 'w') as f:
            f.write(json.dumps(predicted_order))
        
        logger.info(f"Saved predicted edge order to {result_path}")

if __name__ == "__main__":
    main()