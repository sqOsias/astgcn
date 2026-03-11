import numpy as np
import pandas as pd
import os
import json
import configparser
from datetime import datetime

class DataPreprocessor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        # 从配置文件加载参数
        data_cfg = self.config['Data']
        train_cfg = self.config['Training']
        
        # 使用专用的 raw 配置项
        self.data_path = data_cfg.get('graph_signal_matrix_filename_raw', './data/PEMS04/pemsd4.npz')
        self.distance_path = data_cfg.get('adj_filename_raw', './data/PEMS04/pemsd4.csv')
        
        self.num_of_vertices = int(data_cfg['num_of_vertices'])
        self.num_for_predict = int(data_cfg['num_for_predict'])
        self.points_per_hour = int(data_cfg['points_per_hour'])
        
        self.num_of_weeks = int(train_cfg['num_of_weeks'])
        self.num_of_days = int(train_cfg['num_of_days'])
        self.num_of_hours = int(train_cfg['num_of_hours'])
        
        # 计算自动 len_input
        self.calculated_len_input = (self.num_of_weeks + self.num_of_days + self.num_of_hours) * self.num_for_predict
        
        self.start_date = pd.Timestamp('2018-01-01')
        self.interval_min = 60 // self.points_per_hour
        self.zscore_clip = 3.0
        self.mean = None
        self.std = None

    def print_report(self, stage, details):
        """打印数据处理报告"""
        print(f"\n{'='*20} {stage} {'='*20}")
        for k, v in details.items():
            print(f"{k:.<30} {v}")
        print(f"{'='*50}\n")

    def load_raw_data(self):
        print(f"Loading data from {self.data_path}")
        data = np.load(self.data_path)['data']
        self.print_report("Raw Data Loaded", {
            "Shape": data.shape,
            "Nodes": data.shape[1],
            "Features": data.shape[2],
            "Time Steps": data.shape[0],
            "Memory Usage": f"{data.nbytes / 1024**2:.2f} MB"
        })
        return data

    def repair_and_normalize(self, data, train_rate=0.6):
        """修复缺失值并执行归一化"""
        # 1. 线性插值修复缺失 (0值视为缺失)
        repaired = data.copy()
        missing_count = np.sum(repaired == 0)
        
        for n in range(data.shape[1]):
            for f in range(data.shape[2]):
                series = pd.Series(repaired[:, n, f]).replace(0, np.nan)
                repaired[:, n, f] = series.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill').values
        
        # 2. Z-Score 归一化 (仅基于训练集)
        train_size = int(data.shape[0] * train_rate)
        train_set = repaired[:train_size, ...]
        self.mean = np.mean(train_set, axis=(0, 1))
        self.std = np.std(train_set, axis=(0, 1))
        
        normalized = (repaired - self.mean) / (self.std + 1e-8)
        
        # 3. 异常值裁剪
        if self.zscore_clip:
            normalized = np.clip(normalized, -self.zscore_clip, self.zscore_clip)
            
        self.print_report("Data Repair & Normalization", {
            "Missing Values Repaired": missing_count,
            "Train Mean (Speed)": f"{self.mean[2]:.4f}",
            "Train Std (Speed)": f"{self.std[2]:.4f}",
            "Z-Score Clip Range": f"[-{self.zscore_clip}, {self.zscore_clip}]"
        })
        return normalized

    def build_adjacency_matrix(self, sigma=500, epsilon=0.1):
        """高斯核邻接矩阵构建"""
        df = pd.read_csv(self.distance_path)
        unique_ids = np.unique(np.concatenate([df.iloc[:, 0].values, df.iloc[:, 1].values]))
        id_map = {id_: i for i, id_ in enumerate(sorted(unique_ids))}
        num_nodes = len(unique_ids)
        
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edge_count = 0
        for _, row in df.iterrows():
            if row.iloc[0] in id_map and row.iloc[1] in id_map:
                i, j = id_map[row.iloc[0]], id_map[row.iloc[1]]
                dist = row.iloc[2]
                weight = np.exp(-(dist**2) / (sigma**2))
                if weight >= epsilon:
                    adj[i, j] = adj[j, i] = weight
                    edge_count += 1
        np.fill_diagonal(adj, 1.0)
        
        self.print_report("Adjacency Matrix Built", {
            "Nodes": num_nodes,
            "Valid Edges (with Gaussian)": edge_count,
            "Sparsity": f"{np.count_nonzero(adj)/adj.size:.2%}",
            "Sigma": sigma,
            "Epsilon": epsilon
        })
        return adj

    def get_time_embeddings(self, num_steps):
        """生成时间位置编码"""
        timestamps = [self.start_date + pd.Timedelta(minutes=self.interval_min * i) for i in range(num_steps)]
        hours = np.array([ts.hour / 23.0 for ts in timestamps])
        days = np.array([ts.dayofweek / 6.0 for ts in timestamps])
        return np.stack([hours, days], axis=-1) # (T, 2)

    def search_data(self, sequence_length, num_of_depend, label_start_idx, num_for_predict, units, points_per_hour):
        """多周期依赖索引搜索逻辑"""
        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None
        return x_idx[::-1]

    def generate_dataset(self, data):
        """生成包含多周期依赖和时间嵌入的数据集"""
        T, N, F = data.shape
        time_emb = self.get_time_embeddings(T) # (T, 2)
        
        # 拼接到特征维度: [Flow, Occ, Speed, Hour, Day]
        time_emb_expanded = np.tile(time_emb[:, np.newaxis, :], (1, N, 1))
        full_data = np.concatenate([data, time_emb_expanded], axis=-1) # (T, N, F+2)
        
        all_samples = []
        for idx in range(T):
            target_start = idx
            if target_start + self.num_for_predict > T: break
            
            indices = []
            if self.num_of_weeks > 0:
                w_idx = self.search_data(T, self.num_of_weeks, target_start, self.num_for_predict, 7*24, self.points_per_hour)
                if not w_idx: continue
                indices.extend(w_idx)
            if self.num_of_days > 0:
                d_idx = self.search_data(T, self.num_of_days, target_start, self.num_for_predict, 24, self.points_per_hour)
                if not d_idx: continue
                indices.extend(d_idx)
            if self.num_of_hours > 0:
                h_idx = self.search_data(T, self.num_of_hours, target_start, self.num_for_predict, 1, self.points_per_hour)
                if not h_idx: continue
                indices.extend(h_idx)
            
            x_sample = np.concatenate([full_data[i:j] for i, j in indices], axis=0) # (T_total, N, F+2)
            x_sample = x_sample.transpose(1, 2, 0) # (N, F+2, T_total)
            target = data[target_start : target_start + self.num_for_predict, :, 2].transpose(1, 0) # (N, T_out)
            all_samples.append((x_sample, target, idx))

        num_samples = len(all_samples)
        train_end = int(num_samples * 0.6)
        val_end = int(num_samples * 0.8)
        
        def package(samples):
            return np.array([s[0] for s in samples]), np.array([s[1] for s in samples]), np.array([s[2] for s in samples])

        train_x, train_y, train_ts = package(all_samples[:train_end])
        val_x, val_y, val_ts = package(all_samples[train_end:val_end])
        test_x, test_y, test_ts = package(all_samples[val_end:])
        
        self.print_report("Final Dataset Summary (Ready for Model)", {
            "Total Samples": num_samples,
            "Input Shape (X)": train_x.shape,
            "Target Shape (Y)": train_y.shape,
            "Calculated len_input": self.calculated_len_input,
            "Input Features": "Flow, Occ, Speed, HourNorm, DayNorm (5 channels)",
            "Train/Val/Test Split": f"{train_x.shape[0]} / {val_x.shape[0]} / {test_x.shape[0]}"
        })
        
        return {
            'train_x': train_x, 'train_target': train_y, 'train_ts': train_ts,
            'val_x': val_x, 'val_target': val_y, 'val_ts': val_ts,
            'test_x': test_x, 'test_target': test_y, 'test_ts': test_ts,
            'mean': self.mean, 'std': self.std
        }

    def save(self, dataset, adj, save_dir='./data/processed/'):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        # 构造带前缀的文件名
        dataset_name = self.config['Data'].get('dataset_name', 'PEMS')
        prefix = f"{dataset_name}_h{self.num_of_hours}d{self.num_of_days}w{self.num_of_weeks}"
        filename = f"{prefix}_train_data.npz"
        
        save_path = os.path.join(save_dir, filename)
        np.savez_compressed(save_path, **dataset)
        
        # 邻接矩阵也可以加前缀
        adj_path = os.path.join(save_dir, f"{prefix}_adj_mat.npy")
        np.save(adj_path, adj)
        
        params = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        with open(os.path.join(save_dir, 'scaler_params.json'), 'w') as f:
            json.dump(params, f)
            
        print(f"All processed data saved to {save_dir}")
        print(f"Main data file: {filename}")
        print(f"Adjacency matrix: {prefix}_adj_mat.npy")

if __name__ == "__main__":
    pre = DataPreprocessor(config_path='./configurations/PEMS04_astgcn.conf')
    raw = pre.load_raw_data()
    norm_data = pre.repair_and_normalize(raw)
    adj = pre.build_adjacency_matrix()
    dataset = pre.generate_dataset(norm_data)
    pre.save(dataset, adj)
