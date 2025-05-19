# config.py
import os
import torch
import logging

class Config:
    # 基本配置
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据配置
    DATA_PATH = "../net_data/fruit_fly_mirrorTn/edgetime/5_edges_day.txt"  # 数据文件路径
    TRAIN_RATIO = 0.8  # 训练集比例
    VAL_RATIO = 0.1    # 验证集比例
    
    # 图神经网络配置
    GNN_HIDDEN_DIM = 64       # GNN隐藏层维度
    GNN_LAYERS = 2            # GNN层数
    GNN_DROPOUT = 0.1         # GNN dropout率
    
    # 对比学习配置
    PROJ_DIM = 32             # 投影头输出维度
    TEMPERATURE = 0.07        # 温度参数
    TIME_THRESHOLD = 1        # 正样本时间阈值（时间差小于等于此值视为正样本）
    NUM_NEG_SAMPLES = 5       # 每个锚点的负样本数量
    
    # 边序判别器配置
    DISCRIMINATOR_HIDDEN = 32  # 判别器隐藏层维度
    
    # 训练配置
    BATCH_SIZE = 128           # 批次大小
    CONTRASTIVE_EPOCHS = 100   # 对比学习训练轮次
    DISCRIMINATOR_EPOCHS = 50  # 判别器训练轮次
    CONTRASTIVE_LR = 0.001     # 对比学习学习率
    DISCRIMINATOR_LR = 0.001   # 判别器学习率
    PATIENCE = 10              # 早停耐心值
    
    # 模型保存配置
    MODEL_DIR = "saved_models"
    CONTRASTIVE_MODEL_PATH = os.path.join(MODEL_DIR, "contrastive_model.pt")
    DISCRIMINATOR_MODEL_PATH = os.path.join(MODEL_DIR, "discriminator_model.pt")
    
    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DIR = "logs"
    LOG_FILE = os.path.join(LOG_DIR, "training.log")
    
    def __init__(self):
        # 创建必要的目录
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            handlers=[
                logging.FileHandler(self.LOG_FILE),
                logging.StreamHandler()
            ]
        )