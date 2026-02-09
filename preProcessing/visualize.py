import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# 从同级目录导入你之前写的预处理类
from DataPreprocessor import DataPreprocessor

# 设置中文字体以优化论文配图（若环境不支持可注释掉）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- 路径补全：基于项目根目录 astgcn ---
# 提示：在终端执行时，请确保在 astgcn 根目录下运行 python preProcessing/visualize.py
RAW_DATA_PATH = './data/PEMS04/PEMS04.npz'    # 修正了拼写，对应磁盘上的 PEMS04.npz
DIST_PATH = './data/PEMS04/distance.csv'
FIG_SAVE_DIR = './fig/'                       # 图像保存至项目根目录下的 fig 文件夹

def run_visualization():
    # 确保保存目录存在
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)
        print(f"create Directory: {FIG_SAVE_DIR}")

    # 1. 初始化预处理类并加载原始数据
    # 对应任务书：深入分析数据结构，包括时间戳、传感器/路段ID等 [cite: 1]
    dp = DataPreprocessor(RAW_DATA_PATH, DIST_PATH)
    raw_data = dp.load_raw_data()
    
    # 2. 生成中间状态数据用于对比效果
    print("Executing interpolation repair and normalization logic...")
    # 插值修复对比：对应开题报告 1.1 修复非随机缺失模式 [cite: 38]
    cleaned_data = dp.repair_data(raw_data)
    # Z-Score 归一化：对应任务书 2，标准化处理量级差异 [cite: 1]
    norm_data = dp.z_score_normalization(cleaned_data)
    # 邻接矩阵构建：对应开题报告 1.1 利用高斯核函数构建路网邻接矩阵 [cite: 38]
    adj_matrix = dp.build_adjacency_matrix()

    # --- 开始绘图 ---

    # 图 1: 插值修复效果对比
    plt.figure(figsize=(12, 5))
    node_id, feat_id = 0, 2  # 选取第一个传感器的速度特征
    time_slice = slice(200, 500) # 选取一段包含波动的数据
    plt.plot(raw_data[time_slice, node_id, feat_id], label='Raw Data (Missing=0)', color='lightgray', linewidth=3) # 加粗灰色线条
    plt.plot(cleaned_data[time_slice, node_id, feat_id], label='Repaired Data', color='blue', linestyle='--', linewidth=1)
    
    # 关键修复：设置 Y 轴范围从 0 开始，这样能看到缺失值
    plt.ylim(bottom=0) 
    plt.title(f"Speed Feature Repair Comparison (Node {node_id})\nNotice: Gray line drops to 0 at missing points")
    plt.xlabel("Time Steps (5 min/step)")
    plt.ylabel("Speed (mph)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(FIG_SAVE_DIR, '01_repair_contrast.png'), dpi=300)

    # 图 2: 归一化后的数据分布
    plt.figure(figsize=(12, 5))
    speed_raw = cleaned_data[:, :, 2].flatten()
    speed_norm = norm_data[:, :, 2].flatten()
    plt.subplot(1, 2, 1)
    sns.histplot(speed_raw[:20000], kde=True, color='skyblue')
    plt.title("Raw Speed Distribution")
    plt.subplot(1, 2, 2)
    sns.histplot(speed_norm[:20000], kde=True, color='salmon')
    plt.title("Normalized Distribution (Z-Score)")
    plt.savefig(os.path.join(FIG_SAVE_DIR, '02_distribution_contrast.png'), dpi=300)

    # 图 3: 路网时空热力图 (展示拥堵演变趋势)
    # 对应开题报告：展现拥堵状态在上下游路段间的传播 [cite: 40]
    plt.figure(figsize=(14, 8))
    # 选取前 100 个节点一整天（288个时间步）的数据
    heatmap_data = cleaned_data[:288, :100, 2].T 
    sns.heatmap(heatmap_data, cmap='RdYlGn', cbar_kws={'label': 'Speed (mph)'})
    plt.title("Spatio-Temporal Speed Evolution (First 100 Nodes)")
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Node Index")
    plt.savefig(os.path.join(FIG_SAVE_DIR, '03_st_heatmap.png'), dpi=300)

    # 图 4: 邻接矩阵局部热力图
    # 展示空间拓扑先验知识 [cite: 38]
    plt.figure(figsize=(10, 8))
    # 稍微调整颜色映射，让非零值更明显
    # 使用 masked array 把 0 值设为白色/透明，只显示有权重的点
    sns.heatmap(adj_matrix[:50, :50], cmap='Blues', vmin=0, vmax=1, linewidths=0.01, linecolor='white')
    plt.title("Adjacency Matrix (Top-Left 50x50 Subgraph)\nDarker blue = Stronger spatial connection")
    plt.savefig(os.path.join(FIG_SAVE_DIR, '04_adj_matrix.png'), dpi=300)

    print(f"visulize finished, all figures saved to: {os.path.abspath(FIG_SAVE_DIR)}")

if __name__ == "__main__":
    run_visualization()