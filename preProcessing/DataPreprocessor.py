import numpy as np
import pandas as pd
import os
import pickle


class DataPreprocessor:
    """
    针对 PeMSD4 数据集的预处理工具类
    功能：数据加载、缺失值分析、插值修复、Z-Score归一化、邻接矩阵构建
    """
    def __init__(self, data_path, distance_path, device='cpu'):
        self.data_path = data_path
        self.distance_path = distance_path
        self.mean = None
        self.std = None
        
    def load_raw_data(self):
        """
        加载 PEMSD04.npz 文件 
        返回维度为 (Time_steps, Nodes, Features) 的张量
        """
        data = np.load(self.data_path)['data']
        # PeMSD4 通常包含 3 个特征: [流量, 占有率, 速度]
        print(f"loading data success,origin dim: {data.shape}")
        return data

    def analyze_missing_values(self, data):
        """
        分析缺失值情况。在 PeMS 数据中，0 值通常代表缺失点 
        """
        total_elements = data.size
        zero_count = np.sum(data == 0)
        missing_rate = (zero_count / total_elements) * 100
        print(f"total elments: {total_elements}")
        print(f"zero account: {zero_count}")
        print(f"Initial missing rate: {missing_rate:.2f}%")
        return missing_rate

    def repair_data(self, data):
        """
        设计插值修复算法 
        对于速度（特征索引2）中的0值，采用线性插值或时序邻域均值修复
        """
        repaired_data = data.copy()
        for node in range(data.shape[1]):
            for feat in range(data.shape[2]):
                series = repaired_data[:, node, feat]
                if np.any(series == 0):
                    # 将0转换为NaN以便使用pandas的插值功能
                    temp_series = pd.Series(series).replace(0, np.nan)
                    # 采用双向线性插值填充单点缺失
                    temp_series = temp_series.interpolate(method='linear', limit_direction='both')
                    # 对于长序列缺失，使用均值填充剩余部分
                    temp_series = temp_series.fillna(temp_series.mean())
                    repaired_data[:, node, feat] = temp_series.values
        print("Data interpolation repair completed")
        return repaired_data

    def z_score_normalization(self, data, train_rate=0.7):
        """
        执行 Z-Score 归一化 
        计算公式: z = (x - mu) / sigma
        注：均值和标准差仅从训练集中提取，防止数据泄露
        """
        train_size = int(data.shape[0] * train_rate)
        train_set = data[:train_size, ...]
        
        self.mean = np.mean(train_set, axis=(0, 1))
        self.std = np.std(train_set, axis=(0, 1))
        
        normalized_data = (data - self.mean) / self.std
        print(f"normalization completed. mean: {self.mean}, std: {self.std}")
        return normalized_data

    def inverse_normalization(self, normalized_data):
        """
        反归一化，用于预测结果展示 
        """
        if self.mean is None or self.std is None:
            raise ValueError("Not normalized, cannot reverse operation")
        return normalized_data * self.std + self.mean

    def build_adjacency_matrix(self, sigma=10, epsilon=0.5):
        """
        根据 distance.csv 构建邻接矩阵 
        采用高斯核函数计算语义相似度
        """
        try:
            df = pd.read_csv(self.distance_path)
            # 关键修复：确保列名正确，通常 PeMS 是 'from', 'to', 'cost'
            # 如果csv没有表头，需要根据实际情况调整，这里假设有表头
            
            # 1. 收集所有唯一的传感器ID
            unique_ids = np.unique(np.concatenate([df['from'].values, df['to'].values]))
            unique_ids.sort() # 排序，确保顺序一致
            
            # 2. 建立 ID -> 索引 的映射字典
            self.id_map = {id_: i for i, id_ in enumerate(unique_ids)}
            num_nodes = len(unique_ids)
            
            print(f"detected {num_nodes} unique sensor nodes")
            
            # 3. 初始化正确的矩阵大小 (N, N)
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            
            # 4. 填充矩阵
            for _, row in df.iterrows():
                # 使用映射后的索引，而不是原始ID
                if row['from'] in self.id_map and row['to'] in self.id_map:
                    i = self.id_map[int(row['from'])]
                    j = self.id_map[int(row['to'])]
                    dist = row['cost']
                    
                    weight = np.exp(- (dist**2) / (sigma**2))
                    # 只有权重大于阈值才保留，防止矩阵过密
                    if weight >= epsilon:
                        adj[i, j] = weight
                        adj[j, i] = weight # 无向图

            # 对角线设为 1 (自己对自己相关性最强)
            np.fill_diagonal(adj, 1.0)
            
            print(f"Adjacency matrix construction completes, dimensions: {adj.shape}，Non-zero element proportion: {np.count_nonzero(adj)/adj.size:.2%}")
            return adj
            
        except Exception as e:
            print(f" Errors in constructing adjacency matrices: {e}")
            return np.zeros((307, 307)) # 返回空矩阵防止程序崩溃

    def generate_task_data(self, data, window_size=12, horizon=12):
        """
        构造监督学习数据集 [cite: 40]
        window_size: 历史时间窗口长度
        horizon: 预测未来时间步长
        """
        x, y = [], []
        for i in range(len(data) - window_size - horizon):
            x.append(data[i : i + window_size, ...])
            # 预测目标通常为速度（索引2）
            y.append(data[i + window_size : i + window_size + horizon, :, 2])
            
        return np.array(x), np.array(y)
    

    # 预处理数据持久化
    def save_processed_data(self, X, Y, adj, save_dir='./data/processed/'):
        """
        将预处理后的数据和元数据保存到本地
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"create Directory: {save_dir}")

        # 1. 保存特征和标签张量 (使用压缩格式以节省空间)
        # 对应任务书：构建时空序列特征数据集 
        np.savez_compressed(
            os.path.join(save_dir, 'train_data.npz'), 
            x=X, 
            y=Y
        )
        
        # 2. 保存邻接矩阵
        # 对应开题报告：构建包含权重信息的路网邻接矩阵 [cite: 38]
        np.save(os.path.join(save_dir, 'adj_mat.npy'), adj)

        # 3. 保存归一化参数 (用于在线推理阶段的反归一化)
        # 对应开题报告：实现从理论算法到软件系统的转化 
        scaler_params = {
            'mean': self.mean,
            'std': self.std
        }
        with open(os.path.join(save_dir, 'scaler_params.pkl'), 'wb') as f:
            pickle.dump(scaler_params, f)
            
        print(f"all processed data saved to: {save_dir}")
        print(f"saved content: feature tensor X, label tensor Y, adjacency matrix Adj, normalization parameters Mean/Std")

# 使用示例
if __name__ == "__main__":
    # 初始化预处理类
    preprocessor = DataPreprocessor(
        data_path='./data/PEMS04/PEMS04.npz', 
        distance_path='./data/PEMS04/distance.csv'
    )
    
    # 1. 加载数据
    raw_data = preprocessor.load_raw_data()
    
    # 2. 缺失值分析
    preprocessor.analyze_missing_values(raw_data)
    
    # 3. 数据修复
    cleaned_data = preprocessor.repair_data(raw_data)
    
    # 4. 归一化
    norm_data = preprocessor.z_score_normalization(cleaned_data)
    
    # 5. 构建图拓扑邻接矩阵
    adj_matrix = preprocessor.build_adjacency_matrix()
    
    # 6. 生成模型输入 (以历史1小时预测未来1小时为例，假设5min采样)
    X, Y = preprocessor.generate_task_data(norm_data, window_size=12, horizon=12)
    print(f"Final input feature shape: {X.shape}, label shape: {Y.shape}")

    # 7. 保存预处理结果
    preprocessor.save_processed_data(X, Y, adj_matrix)