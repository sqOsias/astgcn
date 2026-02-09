import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 路径设置
ADJ_PATH = './data/processed/adj_mat.npy' # 确保你之前运行过保存脚本
SAVE_PATH = './fig/05_local_adj_zoom.png'

def plot_local_zoom(n_range=15):
    if not os.path.exists(ADJ_PATH):
        print(f"错误：未在 {ADJ_PATH} 找到邻接矩阵文件，请确认预处理脚本已运行并保存。")
        return
    
    # 加载已构建好的邻接矩阵 
    adj = np.load(ADJ_PATH)
    
    # 选取前 n_range 个节点进行观察
    local_adj = adj[:n_range, :n_range]
    
    plt.figure(figsize=(12, 10))
    
    # annot=True: 在每个格子里显示具体的权重数值
    # fmt=".2f": 保留两位小数
    # linewidths=0.5: 给小格子加一个淡淡的边框线
    sns.heatmap(local_adj, 
                annot=True, 
                fmt=".2f", 
                cmap='Blues', 
                cbar_kws={'label': 'Spatial Weight (Gaussian Kernel)'},
                linewidths=0.5,
                linecolor='#f0f0f0')
    
    plt.title(f"Local Adjacency Matrix Zoom-in (Nodes 0-{n_range-1})\nSpatial Correlation based on Gaussian Kernel", fontsize=14)
    plt.xlabel("Sensor Index", fontsize=12)
    plt.ylabel("Sensor Index", fontsize=12)
    
    if not os.path.exists('./fig/'):
        os.makedirs('./fig/')
        
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"局部放大图已生成：{os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    plot_local_zoom(15) # 15x15 的尺寸最适合在论文中展示数值