import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EDAAnalyzer:
    def __init__(self, raw_data_path, distance_path, save_dir='./fig/eda/'):
        self.raw_data_path = raw_data_path
        self.distance_path = distance_path
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def load_data(self):
        print("Loading raw data...")
        self.data = np.load(self.raw_data_path)['data']
        # data shape: (T, N, F) -> (16992, 307, 3)
        # F: Flow, Occupy, Speed
        self.speed_data = self.data[:, :, 2] # (T, N)
        self.df_dist = pd.read_csv(self.distance_path)
        
    def plot_speed_distribution(self):
        """1. 速度分布直方图"""
        print("Plotting speed distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(self.speed_data.flatten(), bins=50, kde=True, color='skyblue')
        plt.title('Vehicle Speed Distribution Histogram (All Nodes)')
        plt.xlabel('Speed (mph)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.save_dir, '01_speed_distribution.png'))
        plt.close()
        
    def plot_daily_pattern(self):
        """2. 速度日变化规律 (早晚高峰)"""
        print("Plotting daily pattern...")
        # 假设数据是 5min 间隔，一天 288 个点
        # PEMS04 2018-01-01 是周一
        steps_per_day = 288
        num_days = self.speed_data.shape[0] // steps_per_day
        
        # Reshape to (Days, Steps_per_day, Nodes)
        daily_speed = self.speed_data[:num_days*steps_per_day, :].reshape(num_days, steps_per_day, -1)
        
        # 平均所有天和所有节点
        avg_daily_profile = np.mean(daily_speed, axis=(0, 2))
        
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(steps_per_day), avg_daily_profile, linewidth=2, color='orange')
        
        # 标记时间轴
        ticks = range(0, 289, 36) # 每3小时
        labels = [f"{h:02d}:00" for h in range(0, 25, 3)]
        plt.xticks(ticks, labels)
        plt.xlim(0, 288)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.title('Average Daily Speed Profile (Aggregated over all nodes)')
        plt.xlabel('Time of Day')
        plt.ylabel('Average Speed (mph)')
        
        # 标注早晚高峰
        # 通常早高峰 7-9点 (84-108步)，晚高峰 17-19点 (204-228步)
        plt.axvspan(84, 108, color='red', alpha=0.1, label='Morning Peak')
        plt.axvspan(204, 228, color='red', alpha=0.1, label='Evening Peak')
        plt.legend()
        
        plt.savefig(os.path.join(self.save_dir, '02_daily_pattern.png'))
        plt.close()

    def plot_acf_pacf(self):
        """3. ACF/PACF 分析"""
        print("Plotting ACF/PACF...")
        # 选取一个代表性节点（例如方差最大的节点，代表变化最剧烈）
        node_idx = np.argmax(np.std(self.speed_data, axis=0))
        sample_series = self.speed_data[:288*3, node_idx] # 取前3天
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        plot_acf(sample_series, lags=288, ax=axes[0], title=f'Autocorrelation (ACF) - Node {node_idx}')
        # 标记显著的滞后
        axes[0].axvline(x=12, color='r', linestyle='--', alpha=0.5, label='1 Hour (12 steps)')
        axes[0].axvline(x=288, color='g', linestyle='--', alpha=0.5, label='1 Day (288 steps)')
        axes[0].legend()
        
        plot_pacf(sample_series, lags=50, ax=axes[1], title=f'Partial Autocorrelation (PACF) - Node {node_idx}')
        axes[1].axvline(x=12, color='r', linestyle='--', alpha=0.5, label='1 Hour')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '03_acf_pacf.png'))
        plt.close()

    def plot_topology(self):
        """4. 节点连通性散点图"""
        print("Plotting topology...")
        # 由于没有经纬度，我们用简单的力导向布局或者只是绘制距离分布
        # 这里绘制距离分布直方图和连通性
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_dist['cost'], bins=30, kde=True, color='purple')
        plt.title('Sensor Distance Distribution (Edge Weights)')
        plt.xlabel('Distance (meters)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.save_dir, '04_distance_distribution.png'))
        plt.close()
        
        # 绘制邻接矩阵稀疏模式
        # 重新计算 adj
        sigma = 500
        unique_ids = np.unique(np.concatenate([self.df_dist['from'].values, self.df_dist['to'].values]))
        id_map = {id_: i for i, id_ in enumerate(unique_ids)}
        num_nodes = len(unique_ids)
        adj = np.zeros((num_nodes, num_nodes))
        
        for _, row in self.df_dist.iterrows():
            if row['from'] in id_map and row['to'] in id_map:
                i = id_map[int(row['from'])]
                j = id_map[int(row['to'])]
                adj[i, j] = 1
                adj[j, i] = 1 # 无向
                
        plt.figure(figsize=(8, 8))
        plt.spy(adj, markersize=1)
        plt.title(f'Adjacency Matrix Sparsity Pattern\n(Nodes: {num_nodes}, Edges: {np.sum(adj)/2:.0f})')
        plt.savefig(os.path.join(self.save_dir, '05_topology_spy.png'))
        plt.close()

    def generate_data_dictionary(self):
        """生成数据字典"""
        info = """
# PEMS04 数据集数据字典

## 1. 流量数据 (PEMS04.npz)
- **文件格式**: NumPy (.npz) key='data'
- **维度**: (16992, 307, 3) -> (时间步, 节点数, 特征数)
- **时间范围**: 2018-01-01 至 2018-02-28 (59天)
- **采样频率**: 5分钟/次 (288次/天)
- **特征通道**:
    - Channel 0: **Flow** (交通流量, 辆/5分钟)
    - Channel 1: **Occupancy** (车道占有率, 0-1)
    - Channel 2: **Speed** (平均速度, 英里/小时 mph)

## 2. 路网拓扑 (pemsd4.csv)
- **文件格式**: CSV
- **字段说明**:
    - `from`: 起始传感器节点ID (Integer)
    - `to`: 终止传感器节点ID (Integer)
    - `cost`: 节点间物理距离 (Float, 单位: 米)
- **说明**: 定义了传感器之间的空间邻接关系，用于构建图卷积网络的邻接矩阵。

## 3. 预处理后数据 (train_data.npz)
- **维度**: (N_samples, 12, 307, 5)
- **新增特征**:
    - Channel 3: **Time of Day** (归一化小时, 0-1)
    - Channel 4: **Day of Week** (归一化星期, 0-1)
        """
        with open(os.path.join(self.save_dir, 'data_dictionary.md'), 'w', encoding='utf-8') as f:
            f.write(info)
        print(f"Data dictionary saved to {os.path.join(self.save_dir, 'data_dictionary.md')}")

if __name__ == "__main__":
    analyzer = EDAAnalyzer(
        raw_data_path='./data/PEMS04/pemsd4.npz',
        distance_path='./data/PEMS04/pemsd4.csv'
    )
    analyzer.load_data()
    analyzer.plot_speed_distribution()
    analyzer.plot_daily_pattern()
    analyzer.plot_acf_pacf()
    analyzer.plot_topology()
    analyzer.generate_data_dictionary()
    print("EDA completed!")
