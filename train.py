import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from prepareData import DataPreprocessor # 引用你的预处理类
from utils import get_dataloader # 引用上面的数据加载
from model import ASTGCN  # 假设你 clone 的模型在这个位置

# --- 配置参数 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DATA_PATH = './data/processed/'

# --- 1. 准备数据 ---
# 注意：如果数据还没预处理，先运行 DataPreprocessor.py，这里直接加载结果
adj_matrix = np.load(os.path.join(DATA_PATH, 'adj_mat.npy'))
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(DEVICE)

train_loader, val_loader, test_loader = get_dataloader(DATA_PATH, BATCH_SIZE)

# 加载归一化参数用于后续反归一化
with open(os.path.join(DATA_PATH, 'scaler_params.pkl'), 'rb') as f:
    scaler_params = pickle.load(f)
    mean, std = scaler_params['mean'], scaler_params['std']
    # 注意：这里的 mean/std 维度可能需要调整以匹配 output

# --- 2. 初始化模型 ---
# 请根据 clone 的代码调整参数
model = ASTGCN(
    nb_block=2, 
    in_channels=3, 
    K=3, 
    nb_chev_filter=64, 
    nb_time_filter=64, 
    time_strides=1, 
    num_for_predict=12, 
    len_input=12, 
    num_of_vertices=307, # PeMSD4 节点数
    adj_mx=adj_matrix
).to(DEVICE)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. 训练循环 ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # batch_x shape: (B, Time, Nodes, Feat) -> 可能需要 permute 为 (B, Nodes, Feat, Time)
        # 根据模型要求调整维度!
        batch_x = batch_x.permute(0, 2, 3, 1).to(DEVICE) 
        batch_y = batch_y.to(DEVICE) # Label 通常是 (B, Nodes, Pred_Time)
        
        optimizer.zero_grad()
        output = model(batch_x) # Output 通常是 (B, Nodes, Pred_Time)
        
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
    # --- 验证步骤 (略) ---
    # 在验证集上评估，保存最佳模型 model.state_dict()