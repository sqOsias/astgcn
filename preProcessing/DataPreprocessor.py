import numpy as np
import pandas as pd
import os
import pickle

"""
数据预处理类，用于加载、分析缺失值、修复缺失值、归一化数据等操作
"""
class DataPreprocessor:
    def __init__(self, data_path, distance_path, device='cpu', interp_method='linear', zscore_clip=None):
        self.data_path = data_path
        self.distance_path = distance_path
        self.mean = None
        self.std = None
        self.interp_method = interp_method
        self.zscore_clip = zscore_clip
        
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
        total_elements = data.size
        zero_count = np.sum(data == 0)
        missing_rate = (zero_count / total_elements) * 100
        print(f"total elments: {total_elements}")
        print(f"zero account: {zero_count}")
        print(f"Initial missing rate: {missing_rate:.2f}%")
        return missing_rate

    """
     数据修复（插值 + Z-Score 去噪）
    """
    def repair_data(self, data):
        repaired_data = data.copy()
        if self.zscore_clip is not None:
            mu = np.nanmean(np.where(repaired_data == 0, np.nan, repaired_data), axis=(0, 1), keepdims=True)
            sigma = np.nanstd(np.where(repaired_data == 0, np.nan, repaired_data), axis=(0, 1), keepdims=True)
            z = (repaired_data - mu) / (sigma + 1e-8)
            mask = np.abs(z) > self.zscore_clip
            repaired_data[mask] = 0
        for node in range(data.shape[1]):
            for feat in range(data.shape[2]):
                series = repaired_data[:, node, feat]
                if np.any(series == 0):
                    temp_series = pd.Series(series).replace(0, np.nan)
                    temp_series = temp_series.interpolate(method=self.interp_method, limit_direction='both')
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

    def build_adjacency_matrix(self, sigma=500, epsilon=0.1):
        """
        根据 distance.csv 构建邻接矩阵 
        采用高斯核函数计算语义相似度
        sigma: 高斯核参数 (建议设为距离标准差或中位数，PEMS04约为250-500)
        epsilon: 阈值，小于此值的权重会被过滤掉
        """
        try:
            if not os.path.exists(self.distance_path):
                print(f"Warning: {self.distance_path} not found. Trying pemsd4.csv...")
                dir_name = os.path.dirname(self.distance_path)
                alt_path = os.path.join(dir_name, 'pemsd4.csv')
                if os.path.exists(alt_path):
                    self.distance_path = alt_path
                    print(f"Using {self.distance_path} instead.")
            
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
        增加了时间特征嵌入：hour_of_day, day_of_week
        """
        # 生成时间特征 (Time Embedding)
        # 假设数据是从 2018-01-01 00:00:00 开始，每5分钟一个点
        # PEMS04: 2018-01-01 to 2018-02-28 (59 days) -> 16992 steps
        start_date = pd.Timestamp('2018-01-01 00:00:00')
        timestamps = [start_date + pd.Timedelta(minutes=5*i) for i in range(len(data))]
        
        # 归一化时间特征 (0-1)
        # hour_of_day: 0-23 -> /23
        # day_of_week: 0-6 -> /6
        # minute_of_hour: 0-55 -> /59
        
        time_features = []
        for ts in timestamps:
            # 简单的归一化时间特征
            t_hour = ts.hour / 23.0
            t_day = ts.dayofweek / 6.0
            # t_min = ts.minute / 59.0 # Optional
            time_features.append([t_hour, t_day])
            
        time_features = np.array(time_features) # (T, 2)
        
        # 将时间特征扩展到所有节点
        # data shape: (T, N, F)
        # time_features shape: (T, 2)
        T, N, F = data.shape
        time_features = np.array(time_features)
        
        # 修复：正确构造 time_features_expanded
        # 先 reshape time_features 到 (T, 1, 2)
        # 然后 tile 到 (T, N, 2)
        time_features_expanded = np.tile(time_features[:, np.newaxis, :], (1, N, 1))
        
        # 合并特征: (T, N, F+2)
        # 注意：这里我们把时间特征拼接到特征维度上
        # 原始特征：Flow, Occupy, Speed
        # 新特征：Flow, Occupy, Speed, HourNorm, DayNorm
        data_with_time = np.concatenate([data, time_features_expanded], axis=2)
        
        print(f"Time embedding added. New feature shape: {data_with_time.shape}")
        
        x, y = [], []
        for i in range(len(data) - window_size - horizon):
            x.append(data_with_time[i : i + window_size, ...])
            # 预测目标通常为速度（索引2）
            y.append(data[i + window_size : i + window_size + horizon, :, 2])
            
        return np.array(x), np.array(y)
    

    # 预处理数据持久化
    def save_processed_data(self, X, Y, adj, save_dir='./data/processed/', split_ratio=(0.6, 0.2, 0.2)):
        """
        将预处理后的数据和元数据保存到本地
        自动执行数据集划分：训练集/验证集/测试集
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"create Directory: {save_dir}")

        # 数据集划分
        samples = X.shape[0]
        train_size = int(samples * split_ratio[0])
        val_size = int(samples * split_ratio[1])
        
        train_x, val_x, test_x = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        train_y, val_y, test_y = Y[:train_size], Y[train_size:train_size+val_size], Y[train_size+val_size:]
        
        print(f"Data Split Summary:")
        print(f"  - Train: {train_x.shape}")
        print(f"  - Val:   {val_x.shape}")
        print(f"  - Test:  {test_x.shape}")

        # 1. 保存特征和标签张量 (兼容 ASTGCN loader 格式)
        # 对应任务书：构建时空序列特征数据集 
        np.savez_compressed(
            os.path.join(save_dir, 'train_data.npz'), 
            x=X, y=Y,  # 保留原始全量数据
            train_x=train_x, train_target=train_y,
            val_x=val_x, val_target=val_y,
            test_x=test_x, test_target=test_y,
            mean=self.mean, std=self.std # 保存归一化参数到同一文件方便读取
        )
        
        # 2. 保存邻接矩阵
        # 对应开题报告：构建包含权重信息的路网邻接矩阵 [cite: 38]
        np.save(os.path.join(save_dir, 'adj_mat.npy'), adj)

        # 3. 保存归一化参数 (用于在线推理阶段的反归一化)
        # 对应开题报告：实现从理论算法到软件系统的转化 
        # 使用 JSON 保存以避免 pickle/numpy 版本兼容性问题
        scaler_params = {
            'mean': self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
            'std': self.std.tolist() if isinstance(self.std, np.ndarray) else self.std
        }
        import json
        with open(os.path.join(save_dir, 'scaler_params.json'), 'w') as f:
            json.dump(scaler_params, f)
            
        print(f"all processed data saved to: {save_dir}")
        print(f"saved content: feature tensor X, label tensor Y, adjacency matrix Adj, normalization parameters Mean/Std (JSON)")

# 使用示例
if __name__ == "__main__":
    # 初始化预处理类
    preprocessor = DataPreprocessor(
        data_path='./data/PEMS04/pemsd4.npz', 
        distance_path='./data/PEMS04/pemsd4.csv'
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
