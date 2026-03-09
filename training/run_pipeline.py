#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASTGCN 模型训练完整流程脚本
功能：执行数据加载、模型训练、验证和测试评估的完整流程
输出：模型权重文件、评估指标 CSV、预测结果 NPZ 文件
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import (
    load_graphdata_channel1, 
    get_adjacency_matrix, 
    compute_val_loss_mstgcn, 
    predict_and_save_results_mstgcn
)
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ASTGCN 训练脚本')
    parser.add_argument('--config', type=str, default='configurations/PEMS04_astgcn.conf',
                        help='配置文件路径')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='是否使用 CPU 训练（默认使用 GPU）')
    return parser.parse_args()

def setup_device(use_cpu=False):
    """设置计算设备"""
    if use_cpu:
        device = torch.device('cpu')
        print("✓ 使用 CPU 进行训练")
    else:
        # 检查 CUDA 可用性
        if torch.cuda.is_available():
            cuda_device = training_config.get('ctx', '0')
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            device = torch.device('cuda:0')
            print(f"✓ 使用 GPU 进行训练：{device}")
        else:
            print("⚠ CUDA 不可用，自动切换到 CPU")
            device = torch.device('cpu')
    return device

if __name__ == "__main__":
    args = parse_args()
    
    # ========== 1. 读取配置 ==========
    print("="*60)
    print("步骤 1: 读取配置文件")
    print("="*60)
    config = configparser.ConfigParser()
    print(f'读取配置文件：{args.config}')
    config.read(args.config)
    
    data_config = config['Data']
    training_config = config['Training']
    
    # 数据参数
    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    id_filename = data_config.get('id_filename', None)
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']
    
    # 模型参数
    model_name = training_config['model_name']
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    batch_size = int(training_config['batch_size'])
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    time_strides = num_of_hours
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    in_channels = int(training_config['in_channels'])
    nb_block = int(training_config['nb_block'])
    K = int(training_config['K'])
    loss_function = training_config['loss_function']
    metric_method = training_config['metric_method']
    missing_value = float(training_config['missing_value'])
    
    print(f"✓ 数据集：{dataset_name}")
    print(f"✓ 节点数：{num_of_vertices}")
    print(f"✓ 输入长度：{len_input} 时间步")
    print(f"✓ 预测长度：{num_for_predict} 时间步")
    
    # ========== 2. 设置设备 ==========
    print("\n" + "="*60)
    print("步骤 2: 设置计算设备")
    print("="*60)
    DEVICE = setup_device(use_cpu=args.cpu)
    
    # ========== 3. 加载数据 ==========
    print("\n" + "="*60)
    print("步骤 3: 加载训练数据")
    print("="*60)
    
    train_loader, train_target_tensor, val_loader, val_target_tensor, \
    test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days, 
        num_of_weeks, DEVICE, batch_size
    )
    
    print(f"✓ 训练集大小：{len(train_loader) * batch_size} 样本")
    print(f"✓ 验证集大小：{len(val_loader) * batch_size} 样本")
    print(f"✓ 测试集大小：{len(test_loader) * batch_size} 样本")
    
    # ========== 4. 构建邻接矩阵 ==========
    print("\n" + "="*60)
    print("步骤 4: 构建路网邻接矩阵")
    print("="*60)
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    print(f"✓ 邻接矩阵形状：{adj_mx.shape}")
    print(f"✓ 非零元素比例：{np.count_nonzero(adj_mx) / adj_mx.size:.2%}")
    
    # ========== 5. 初始化模型 ==========
    print("\n" + "="*60)
    print("步骤 5: 初始化 ASTGCN 模型")
    print("="*60)
    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, 
                     nb_time_filter, time_strides, adj_mx, num_for_predict, 
                     len_input, num_of_vertices)
    
    # 统计参数量
    total_param = sum(p.numel() for p in net.parameters())
    trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"✓ 模型总参数量：{total_param:,}")
    print(f"✓ 可训练参数量：{trainable_param:,}")
    
    # ========== 6. 设置损失函数和优化器 ==========
    print("\n" + "="*60)
    print("步骤 6: 设置损失函数和优化器")
    print("="*60)
    
    masked_flag = 0
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse
        masked_flag = 1
        print("✓ 损失函数：Masked MSE")
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
        print("✓ 损失函数：Masked MAE")
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
        print("✓ 损失函数：MAE (L1 Loss)")
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
        print("✓ 损失函数：RMSE (MSE Loss)")
    else:
        raise ValueError(f"不支持的损失函数：{loss_function}")
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print(f"✓ 优化器：Adam (lr={learning_rate})")
    
    # ========== 7. 准备训练目录 ==========
    print("\n" + "="*60)
    print("步骤 7: 准备模型保存目录")
    print("="*60)
    folder_dir = f'{model_name}_h{num_of_hours}d{num_of_days}w{num_of_weeks}_channel{in_channels}_{learning_rate:.6f}'
    params_path = os.path.join('experiments', dataset_name, folder_dir)
    
    if start_epoch == 0 and not os.path.exists(params_path):
        os.makedirs(params_path)
        print(f"✓ 创建参数目录：{params_path}")
    elif start_epoch == 0 and os.path.exists(params_path):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print(f"✓ 重新创建参数目录：{params_path}")
    elif start_epoch > 0 and os.path.exists(params_path):
        print(f"✓ 从断点继续训练：{params_path}")
    else:
        raise SystemExit('模型类型错误！')
    
    # TensorBoard 日志
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(f"✓ TensorBoard 日志目录：{params_path}")
    
    # ========== 8. 加载预训练权重（如果有） ==========
    if start_epoch > 0:
        params_filename = os.path.join(params_path, f'epoch_{start_epoch}.params')
        net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
        print(f"✓ 加载权重：{params_filename}")
    
    # ========== 9. 开始训练 ==========
    print("\n" + "="*60)
    print("步骤 9: 开始训练循环")
    print("="*60)
    
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()
    global_step = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time()
        
        # --- 9.1 验证集评估 ---
        net.train(False)
        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, 
                                               masked_flag, missing_value, sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, 
                                               masked_flag, missing_value, sw, epoch)
        
        epoch_time = time() - epoch_start_time
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  验证集损失：{val_loss:.4f}')
        print(f'  用时：{epoch_time:.2f}s')
        
        # --- 9.2 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            params_filename = os.path.join(params_path, f'epoch_{epoch}.params')
            torch.save(net.state_dict(), params_filename)
            print(f'  ✓ 保存最佳模型：{params_filename}')
            print(f'  ✓ 最佳验证损失：{best_val_loss:.4f}')
        
        # --- 9.3 训练集训练 ---
        net.train(True)
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data
            
            optimizer.zero_grad()
            outputs = net(encoder_inputs)
            
            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            
            if global_step % 100 == 0:
                print(f'  Step {global_step}: 训练损失 = {training_loss:.4f}')
    
    # ========== 10. 测试集评估 ==========
    print("\n" + "="*60)
    print("步骤 10: 在测试集上评估最佳模型")
    print("="*60)
    
    print(f'最佳 Epoch: {best_epoch}')
    print(f'最佳验证损失：{best_val_loss:.4f}')
    
    # 加载最佳模型
    best_params_filename = os.path.join(params_path, f'epoch_{best_epoch}.params')
    net.load_state_dict(torch.load(best_params_filename, map_location=DEVICE))
    print(f'✓ 加载最佳模型：{best_params_filename}')
    
    # 在测试集上评估
    predict_and_save_results_mstgcn(net, test_loader, test_target_tensor, 
                                    best_epoch, metric_method, _mean, _std, 
                                    params_path, 'test')
    
    # ========== 11. 生成训练报告 ==========
    print("\n" + "="*60)
    print("步骤 11: 生成训练总结报告")
    print("="*60)
    
    total_training_time = time() - start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    report = f"""
{'='*60}
训练完成报告
{'='*60}
模型名称：{model_name}
数据集：{dataset_name}
设备：{DEVICE}

训练参数:
  - Epochs: {epochs}
  - Batch Size: {batch_size}
  - Learning Rate: {learning_rate}
  - 输入长度：{len_input}
  - 预测长度：{num_for_predict}

训练结果:
  - 最佳 Epoch: {best_epoch}
  - 最佳验证损失：{best_val_loss:.4f}
  - 总训练时间：{hours}h {minutes}m {seconds}s

输出文件:
  - 模型权重：{best_params_filename}
  - 评估指标：{params_path}/metrics_results_test.csv
  - 预测结果：{params_path}/output_epoch_{best_epoch}_test.npz
  - 详细对比：{params_path}/predictions_compare_test.csv
  - TensorBoard 日志：{params_path}/runs/
{'='*60}
"""
    print(report)
    
    # 保存训练报告
    report_path = os.path.join(params_path, 'training_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ 训练报告已保存：{report_path}")
    
    print("\n🎉 训练完成！")
