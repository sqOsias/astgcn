import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class TrafficDataset(Dataset):
    def __init__(self, x, y):
        # 确保数据是 float32 类型，PyTorch 默认使用 float32
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # ASTGCN 等图网络通常需要的输入维度可能不同
        # 你的预处理输出 X 维度: (Samples, Time_steps, Nodes, Features)
        # 如果 ASTGCN 需要 (Samples, Features, Time_steps, Nodes) 或其他，需在这里 permute
        # 假设 ASTGCN 输入要求: (Batch, Node, Time, Feature) -> 根据具体实现调整
        return self.x[idx], self.y[idx]

def get_dataloader(data_path, batch_size=64, train_ratio=0.7, val_ratio=0.1):
    # 1. 加载预处理好的数据
    data = np.load(os.path.join(data_path, 'train_data.npz'))
    X, Y = data['x'], data['y']
    
    # 2. 按时间顺序划分数据集 (切忌 shuffle 打乱时间顺序！)
    len_data = X.shape[0]
    train_len = int(len_data * train_ratio)
    val_len = int(len_data * val_ratio)
    
    train_x, train_y = X[:train_len], Y[:train_len]
    val_x, val_y = X[train_len:train_len+val_len], Y[train_len:train_len+val_len]
    test_x, test_y = X[train_len+val_len:], Y[train_len+val_len:]
    
    print(f"Train size: {train_x.shape}, Val size: {val_x.shape}, Test size: {test_x.shape}")

    # 3. 构建 Dataset
    train_set = TrafficDataset(train_x, train_y)
    val_set = TrafficDataset(val_x, val_y)
    test_set = TrafficDataset(test_x, test_y)

    # 4. 构建 DataLoader (训练集可以 shuffle，验证/测试集不行)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader