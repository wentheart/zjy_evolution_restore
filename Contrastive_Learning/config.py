# config.py
import os
import torch
import logging
import datetime
import argparse

class Config:
    # 基本配置
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据集配置
    DATASETS = [
        "ants",
        "bacteria_mirrorTn",
        "Coach",
        "complex_networks",
        "Ferry",
        "fruit_fly_mirrorTn",
        "thermodynamics",
        "weaver",
        "worm_mirrorTn"
    ]
    CURRENT_DATASET = "fruit_fly_mirrorTn"  # 默认数据集
    
    # 数据配置
    @property
    def DATA_PATH(self):
        return f"../net_data/{self.CURRENT_DATASET}/edgetime/{self.CURRENT_DATASET}.txt"
    
    TRAIN_RATIO = 0.4  # 训练集比例
    VAL_RATIO = 0.1    # 验证集比例
    
    # 图神经网络配置
    GNN_HIDDEN_DIM = 64       # GNN隐藏层维度
    GNN_LAYERS = 2            # GNN层数
    GNN_DROPOUT = 0.1         # GNN dropout率
    
    # 对比学习配置
    PROJ_DIM = 32             # 投影头输出维度
    TEMPERATURE = 0.07        # 温度参数
    TIME_THRESHOLD = 0.5        # 正样本时间阈值（时间差小于等于此值视为正样本）
    NUM_NEG_SAMPLES = 5       # 每个锚点的负样本数量
    
    # 边序判别器配置
    DISCRIMINATOR_HIDDEN = 32  # 判别器隐藏层维度
    
    # 训练配置
    BATCH_SIZE = 128           # 批次大小
    CONTRASTIVE_EPOCHS = 100   # 对比学习训练轮次
    DISCRIMINATOR_EPOCHS = 100  # 判别器训练轮次
    CONTRASTIVE_LR = 0.001     # 对比学习学习率
    DISCRIMINATOR_LR = 0.001   # 判别器学习率
    PATIENCE = 10              # 早停耐心值
    
    # 模型保存配置
    MODEL_DIR = "./saved_models"
    
    @property
    def CONTRASTIVE_MODEL_PATH(self):
        return os.path.join(self.MODEL_DIR, f"{self.CURRENT_DATASET}_contrastive_model.pt")
    
    @property
    def DISCRIMINATOR_MODEL_PATH(self):
        return os.path.join(self.MODEL_DIR, f"{self.CURRENT_DATASET}_discriminator_model.pt")
    
    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DIR = "./logs"
    
    @property
    def LOG_FILE(self):

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.LOG_DIR, f"{self.CURRENT_DATASET}_{timestamp}.log")
    
    def update_dataset(self, dataset_name):
        """更新当前数据集"""
        if dataset_name in self.DATASETS:
            self.CURRENT_DATASET = dataset_name
            return True
        else:
            return False
    
    def __init__(self):
        # 创建必要的目录
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)


# 全局配置实例
config = Config()


def setup_logging(config):
    """设置日志配置"""
    # 配置日志
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


def add_dataset_arg(parser):
    """为参数解析器添加数据集选择参数"""
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=Config.DATASETS,
        default=config.CURRENT_DATASET,
        help='选择要使用的数据集'
    )
    return parser