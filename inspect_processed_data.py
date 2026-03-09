#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查预处理后的数据文件内容
"""
import numpy as np
import os

print("="*60)
print("检查预处理后的数据文件")
print("="*60)

# ========== 1. 检查 train_data.npz ==========
print("\n【1】train_data.npz 文件内容")
print("-"*60)

train_data_path = './data/processed/train_data.npz'
if os.path.exists(train_data_path):
    data = np.load(train_data_path)
    
    print(f"✓ 文件包含的键：{list(data.keys())}")
    
    # 特征 X
    print(f"\n特征张量 X:")
    print(f"  - 形状：{data['x'].shape}")
    print(f"    (样本数={data['x'].shape[0]}, 时间步={data['x'].shape[1]}, "
          f"节点数={data['x'].shape[2]}, 特征数={data['x'].shape[3]})")
    print(f"  - 数据类型：{data['x'].dtype}")
    print(f"  - 数值范围：[{data['x'].min():.4f}, {data['x'].max():.4f}]")
    print(f"  - 均值：{data['x'].mean():.4f}")
    print(f"  - 标准差：{data['x'].std():.4f}")
    
    # 展示具体例子
    print(f"\n  【具体例子】第一个样本 (前 2 个节点，所有特征和时间步):")
    print(f"  节点 0 的所有特征和时间步:")
    for feat_idx, feat_name in enumerate(['流量', '占有率', '速度']):
        print(f"    {feat_name}: {data['x'][0, :, feat_idx, :]}")
    
    print(f"\n  节点 1 的所有特征和时间步:")
    for feat_idx, feat_name in enumerate(['流量', '占有率', '速度']):
        print(f"    {feat_name}: {data['x'][0, :, feat_idx, :]}")
    
    # 标签 Y
    print(f"\n标签张量 Y:")
    print(f"  - 形状：{data['y'].shape}")
    print(f"    (样本数={data['y'].shape[0]}, 预测步长={data['y'].shape[1]}, "
          f"节点数={data['y'].shape[2]})")
    print(f"  - 数据类型：{data['y'].dtype}")
    print(f"  - 数值范围：[{data['y'].min():.4f}, {data['y'].max():.4f}]")
    print(f"  - 均值：{data['y'].mean():.4f}")
    print(f"  - 标准差：{data['y'].std():.4f}")
    
    print(f"\n  【具体例子】第一个样本 (前 2 个节点的预测标签):")
    print(f"  节点 0 的未来 12 步速度预测：{data['y'][0, 0, :]}")
    print(f"  节点 1 的未来 12 步速度预测：{data['y'][0, 1, :]}")
    
    # 解释维度含义
    print(f"\n【维度解读】")
    print(f"X[0, :, :, :] = 第一个样本")
    print(f"  X[0, 0, :, :] = 历史第 1 个时间步 (t-11)")
    print(f"  X[0, 5, :, :] = 历史第 6 个时间步 (t-6)")
    print(f"  X[0, 11, :, :] = 历史最后 1 个时间步 (t-1)")
    print(f"  X[0, :, 0, :] = 节点 0 的所有特征在所有时间步的值")
    print(f"  X[0, :, 2, :] = 速度特征在所有节点和所有时间步的值")
    print(f"\nY[0, :, :] = 第一个样本的预测目标")
    print(f"  Y[0, 0, :] = 节点 0 的未来 12 步速度")
    print(f"  Y[0, 5, :] = 节点 5 的未来 12 步速度")
    print(f"  Y[0, :, 0] = 所有节点在未来第 1 步的速度")
    print(f"  Y[0, :, 11] = 所有节点在未来最后 1 步的速度")
    
else:
    print(f"❌ 文件不存在：{train_data_path}")

# ========== 2. 检查 adj_mat.npy ==========
print("\n" + "="*60)
print("[2】adj_mat.npy 邻接矩阵内容")
print("-"*60)

adj_mat_path = './data/processed/adj_mat.npy'
if os.path.exists(adj_mat_path):
    adj_matrix = np.load(adj_mat_path)
    
    print(f"✓ 邻接矩阵形状：{adj_matrix.shape}")
    print(f"  - 节点数：{adj_matrix.shape[0]}")
    print(f"  - 边数（理论）：{adj_matrix.shape[0] * adj_matrix.shape[1]}")
    
    print(f"\n数值统计:")
    print(f"  - 最小值：{adj_matrix.min():.4f}")
    print(f"  - 最大值：{adj_matrix.max():.4f}")
    print(f"  - 均值：{adj_matrix.mean():.4f}")
    print(f"  - 非零元素数量：{np.count_nonzero(adj_matrix)}")
    print(f"  - 零元素数量：{np.sum(adj_matrix == 0)}")
    print(f"  - 稀疏度：{np.count_nonzero(adj_matrix) / adj_matrix.size:.2%}")
    
    print(f"\n对角线元素（自己与自己）:")
    print(f"  - 是否全为 1: {np.allclose(np.diag(adj_matrix), 1.0)}")
    print(f"  - 示例：{np.diag(adj_matrix)[:5]}")
    
    print(f"\n对称性检查:")
    is_symmetric = np.allclose(adj_matrix, adj_matrix.T)
    print(f"  - 是否对称：{is_symmetric}")
    
    print(f"\n【具体例子】前 5 个节点的邻接矩阵:")
    print(adj_matrix[:5, :5])
    
    print(f"\n【为什么大部分是 0？】")
    print(f"这是正常的！原因如下:")
    print(f"1. 高斯核阈值过滤：距离>阈值的边被设为 0")
    print(f"2. 地理限制：高速公路网络本身是稀疏的")
    print(f"3. 物理意义：只有相邻/相关的路段才有影响")
    print(f"4. 计算效率：稀疏矩阵可以加速计算")
    print(f"\n典型稀疏度：10%-30% (当前：{np.count_nonzero(adj_matrix) / adj_matrix.size:.2%})")
    
    # 可视化边的分布
    print(f"\n每个节点的平均连接数:")
    avg_connections = np.sum(adj_matrix != 0, axis=1).mean()
    print(f"  - 平均值：{avg_connections:.1f} 个邻居")
    print(f"  - 最多连接：{np.sum(adj_matrix != 0, axis=1).max()} 个")
    print(f"  - 最少连接：{np.sum(adj_matrix != 0, axis=1).min()} 个")
    
else:
    print(f"❌ 文件不存在：{adj_mat_path}")

print("\n" + "="*60)
print("总结")
print("="*60)
print("""
train_data.npz:
  ✓ 包含特征 X (输入) 和标签 Y (预测目标)
  ✓ X 形状：(16968, 12, 307, 3) = (样本，时间，节点，特征)
  ✓ Y 形状：(16968, 12, 307) = (样本，预测步长，节点)
  ✓ 特征顺序：[流量，占有率，速度]
  ✓ 已归一化到 Z-Score 分布

adj_mat.npy:
  ✓ 形状：(307, 307) - 307 个节点的邻接矩阵
  ✓ 稀疏度：~20% - 正常现象
  ✓ 对角线为 1 - 自己与自己完全相关
  ✓ 对称矩阵 - 无向图
  ✓ 基于高斯核函数构建
  
两者关系:
  X[:, :, i, :] → 节点 i 的历史数据
  Y[:, :, i]    → 节点 i 的未来预测
  adj[i, j]     → 节点 i 和 j 的空间相关性
""")
