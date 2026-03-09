#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型性能评估与验证脚本
功能：加载训练好的模型，在测试集上进行评估，验证结果可靠性
输出：详细的评估报告、可视化图表、验证结果
"""

import torch
import numpy as np
import os
import argparse
import configparser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lib.utils import load_graphdata_channel1, get_adjacency_matrix
from model.ASTGCN_r import make_model
from lib.metrics import masked_mape_np, masked_mae_test, masked_rmse_test
import csv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ASTGCN 模型评估脚本')
    parser.add_argument('--config', type=str, default='configurations/PEMS04_astgcn.conf',
                        help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重文件路径')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='是否使用 CPU 进行评估')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation_results',
                        help='评估结果输出目录')
    return parser.parse_args()


def setup_device(use_cpu=False):
    """设置计算设备"""
    if use_cpu:
        device = torch.device('cpu')
        print("✓ 使用 CPU 进行评估")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ 使用 GPU 进行评估：{device}")
        else:
            print("⚠ CUDA 不可用，自动切换到 CPU")
            device = torch.device('cpu')
    return device


def load_model(config, device, model_path):
    """加载模型"""
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    
    # 提取配置参数
    data_config = config['Data']
    training_config = config['Training']
    
    num_of_vertices = int(data_config['num_of_vertices'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    
    in_channels = int(training_config['in_channels'])
    nb_block = int(training_config['nb_block'])
    K = int(training_config['K'])
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    time_strides = int(training_config['time_strides'])
    
    # 加载邻接矩阵
    adj_filename = data_config['adj_filename']
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, 
                                     data_config.get('id_filename', None))
    
    # 构建模型
    net = make_model(device, nb_block, in_channels, K, nb_chev_filter,
                     nb_time_filter, time_strides, adj_mx, num_for_predict,
                     len_input, num_of_vertices)
    
    # 加载权重
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ 模型加载成功：{model_path}")
    else:
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    
    # 设置为评估模式
    net.eval()
    
    # 统计参数量
    total_param = sum(p.numel() for p in net.parameters())
    trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"✓ 模型总参数量：{total_param:,}")
    print(f"✓ 可训练参数量：{trainable_param:,}")
    
    return net


def evaluate_on_testset(net, test_loader, test_target_tensor, device, _mean, _std):
    """在测试集上进行评估"""
    print("\n" + "="*60)
    print("执行预测")
    print("="*60)
    
    net.train(False)
    predictions = []
    inputs = []
    targets = []
    
    with torch.no_grad():
        loader_length = len(test_loader)
        
        for batch_index, batch_data in enumerate(test_loader):
            encoder_inputs, labels = batch_data
            
            outputs = net(encoder_inputs)
            
            predictions.append(outputs.detach().cpu().numpy())
            inputs.append(encoder_inputs[:, :, 0:1].cpu().numpy())
            targets.append(labels.cpu().numpy())
            
            if batch_index % 50 == 0:
                print(f'预测进度：{batch_index + 1}/{loader_length} batches')
        
        # 合并所有 batch
        prediction = np.concatenate(predictions, axis=0)
        input_data = np.concatenate(inputs, axis=0)
        target_data = np.concatenate(targets, axis=0)
        
        print(f'\n✓ 预测完成')
        print(f'  - 输入形状：{input_data.shape}')
        print(f'  - 预测形状：{prediction.shape}')
        print(f'  - 真实值形状：{target_data.shape}')
        
        # 反归一化
        print('\n反归一化处理...')
        prediction_denorm = prediction * _std.numpy() + _mean.numpy()
        target_denorm = target_data * _std.numpy() + _mean.numpy()
        input_denorm = input_data * _std.numpy() + _mean.numpy()
        
        print('✓ 反归一化完成')
        
        return prediction_denorm, target_denorm, input_denorm


def calculate_metrics(prediction, target):
    """计算评估指标"""
    print("\n" + "="*60)
    print("计算评估指标")
    print("="*60)
    
    # 整体指标
    mae_all = mean_absolute_error(target.flatten(), prediction.flatten())
    rmse_all = mean_squared_error(target.flatten(), prediction.flatten()) ** 0.5
    mape_all = masked_mape_np(target.flatten(), prediction.flatten(), 0) * 100
    
    print(f"\n整体指标:")
    print(f"  - MAE: {mae_all:.4f}")
    print(f"  - RMSE: {rmse_all:.4f}")
    print(f"  - MAPE: {mape_all:.2f}%")
    
    # 每个预测步长的指标
    prediction_length = prediction.shape[2]
    step_metrics = []
    
    print(f"\n各预测步长指标:")
    for i in range(prediction_length):
        mae_step = mean_absolute_error(target[:, :, i], prediction[:, :, i])
        rmse_step = mean_squared_error(target[:, :, i], prediction[:, :, i]) ** 0.5
        mape_step = masked_mape_np(target[:, :, i], prediction[:, :, i], 0) * 100
        
        step_metrics.append({
            'step': i + 1,
            'mae': mae_step,
            'rmse': rmse_step,
            'mape': mape_step
        })
        
        print(f"  Step {i+1:2d}: MAE={mae_step:.4f}, RMSE={rmse_step:.4f}, MAPE={mape_step:.2f}%")
    
    return {
        'overall': {'mae': mae_all, 'rmse': rmse_all, 'mape': mape_all},
        'per_step': step_metrics
    }


def visualize_results(prediction, target, input_data, output_dir, metrics):
    """可视化评估结果"""
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 预测步长指标趋势图
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = [m['step'] for m in metrics['per_step']]
    mae_values = [m['mae'] for m in metrics['per_step']]
    rmse_values = [m['rmse'] for m in metrics['per_step']]
    mape_values = [m['mape'] for m in metrics['per_step']]
    
    x = np.arange(len(steps))
    width = 0.25
    
    ax.bar(x - width, mae_values, width, label='MAE', color='skyblue')
    ax.bar(x, rmse_values, width, label='RMSE', color='lightcoral')
    ax.bar(x + width, mape_values, width, label='MAPE (%)', color='lightgreen')
    
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Model Performance Metrics by Prediction Step', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}' for s in steps])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_step.png'), dpi=300)
    print(f"✓ 保存指标趋势图：{os.path.join(output_dir, 'metrics_by_step.png')}")
    plt.close()
    
    # 2. 预测值 vs 真实值对比图（样本选择）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选择前 4 个节点进行展示
    sample_nodes = [0, 1, 2, 3]
    sample_batch = 0
    sample_timestep = 0  # 第一个预测步长
    
    for idx, node_idx in enumerate(sample_nodes):
        ax = axes[idx // 2, idx % 2]
        
        # 提取该节点在所有样本上的预测和真实值
        pred_vals = prediction[sample_batch, node_idx, :, sample_timestep]
        true_vals = target[sample_batch, node_idx, :, sample_timestep]
        
        time_steps = range(len(pred_vals))
        ax.plot(time_steps, pred_vals, 'r-', linewidth=1.5, label='Prediction', alpha=0.7)
        ax.plot(time_steps, true_vals, 'b-', linewidth=1.5, label='Ground Truth', alpha=0.7)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Speed')
        ax.set_title(f'Node {node_idx} - Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_truth_sample.png'), dpi=300)
    print(f"✓ 保存预测对比图：{os.path.join(output_dir, 'prediction_vs_truth_sample.png')}")
    plt.close()
    
    # 3. 散点图（预测值 vs 真实值）
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 随机采样 1000 个点
    n_samples = min(1000, prediction.size)
    indices = np.random.choice(prediction.size, n_samples, replace=False)
    
    pred_flat = prediction.flatten()[indices]
    true_flat = target.flatten()[indices]
    
    ax.scatter(true_flat, pred_flat, alpha=0.5, s=10, c='blue')
    
    # 添加理想拟合线 y=x
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y=x)')
    
    # 计算 R²
    r2 = r2_score(true_flat, pred_flat)
    
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title(f'Prediction vs Ground Truth (R² = {r2:.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300)
    print(f"✓ 保存散点图：{os.path.join(output_dir, 'scatter_plot.png')}")
    plt.close()
    
    # 4. 误差分布直方图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    errors = (prediction - target).flatten()
    ax.hist(errors, bins=100, color='purple', edgecolor='black', alpha=0.7)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Error (Prediction - Ground Truth)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    print(f"✓ 保存误差分布图：{os.path.join(output_dir, 'error_distribution.png')}")
    plt.close()
    
    print(f"\n✓ 所有图表已保存至：{output_dir}")


def save_evaluation_report(metrics, output_dir):
    """保存评估报告"""
    print("\n" + "="*60)
    print("保存评估报告")
    print("="*60)
    
    report_path = os.path.join(output_dir, 'evaluation_report.csv')
    
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入整体指标
        writer.writerow(['Overall Metrics', '', '', ''])
        writer.writerow(['Metric', 'Value', '', ''])
        writer.writerow(['MAE', metrics['overall']['mae'], '', ''])
        writer.writerow(['RMSE', metrics['overall']['rmse'], '', ''])
        writer.writerow(['MAPE (%)', metrics['overall']['mape'], '', ''])
        writer.writerow([])
        
        # 写入各步长指标
        writer.writerow(['Per-Step Metrics', '', '', ''])
        writer.writerow(['Step', 'MAE', 'RMSE', 'MAPE (%)'])
        for m in metrics['per_step']:
            writer.writerow([m['step'], m['mae'], m['rmse'], m['mape']])
    
    print(f"✓ 评估报告已保存：{report_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("ASTGCN 模型性能评估与验证")
    print("="*60)
    
    # ========== 1. 读取配置 ==========
    print("\n步骤 1: 读取配置文件")
    config = configparser.ConfigParser()
    print(f'读取配置文件：{args.config}')
    config.read(args.config)
    
    # ========== 2. 设置设备 ==========
    print("\n步骤 2: 设置计算设备")
    DEVICE = setup_device(use_cpu=args.cpu)
    
    # ========== 3. 加载模型 ==========
    print("\n步骤 3: 加载预训练模型")
    net = load_model(config, DEVICE, args.model_path)
    
    # ========== 4. 加载测试数据 ==========
    print("\n步骤 4: 加载测试数据")
    
    data_config = config['Data']
    training_config = config['Training']
    
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    num_of_hours = int(training_config['num_of_hours'])
    num_of_days = int(training_config['num_of_days'])
    num_of_weeks = int(training_config['num_of_weeks'])
    batch_size = int(training_config['batch_size'])
    
    _, _, test_loader, test_target_tensor, _, _, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days,
        num_of_weeks, DEVICE, batch_size, shuffle=False
    )
    
    print(f"✓ 测试集大小：{len(test_loader) * batch_size} 样本")
    
    # ========== 5. 执行预测 ==========
    print("\n步骤 5: 在测试集上执行预测")
    prediction, target, input_data = evaluate_on_testset(
        net, test_loader, test_target_tensor, DEVICE, _mean, _std
    )
    
    # ========== 6. 计算评估指标 ==========
    print("\n步骤 6: 计算评估指标")
    metrics = calculate_metrics(prediction, target)
    
    # ========== 7. 可视化结果 ==========
    print("\n步骤 7: 生成可视化图表")
    visualize_results(prediction, target, input_data, args.output_dir, metrics)
    
    # ========== 8. 保存评估报告 ==========
    print("\n步骤 8: 保存评估报告")
    save_evaluation_report(metrics, args.output_dir)
    
    # ========== 9. 总结 ==========
    print("\n" + "="*60)
    print("🎉 评估完成！")
    print("="*60)
    print(f"\n评估结果已保存至：{args.output_dir}")
    print(f"  - 评估报告：evaluation_report.csv")
    print(f"  - 指标图表：metrics_by_step.png")
    print(f"  - 预测对比：prediction_vs_truth_sample.png")
    print(f"  - 散点图：scatter_plot.png")
    print(f"  - 误差分布：error_distribution.png")
    
    print(f"\n最终性能指标:")
    print(f"  - MAE:  {metrics['overall']['mae']:.4f}")
    print(f"  - RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"  - MAPE: {metrics['overall']['mape']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
