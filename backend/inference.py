# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
from model.ASTGCN_r import make_model
from lib.utils import get_adjacency_matrix, scaled_Laplacian, cheb_polynomial


class ASTGCNInference:
    """
    ASTGCN 推理引擎类
    支持单条和批量预测
    """
    
    def __init__(self, config_dict, model_path, device):
        """
        初始化推理引擎
        
        Args:
            config_dict: 配置字典 (包含模型参数)
            model_path: 训练好的模型权重文件路径
            device: 计算设备 ('cuda:0' 或 'cpu')
        """
        self.device = torch.device(device)
        self.config = config_dict
        
        # 加载配置参数
        self.data_config = config_dict.get('data', {})
        self.training_config = config_dict.get('training', {})

        self.num_of_vertices = self.data_config['num_of_vertices']
        self.in_channels = self.training_config['in_channels']
        self.nb_block = self.training_config['nb_block']
        self.K = self.training_config['K']
        self.nb_chev_filter = self.training_config['nb_chev_filter']
        self.nb_time_filter = self.training_config['nb_time_filter']
        self.time_strides = self.training_config['time_strides']
        self.num_for_predict = self.data_config['num_for_predict']
        self.len_input = self.data_config['len_input']
        
        # 加载邻接矩阵
        self.adj_mx, _ = get_adjacency_matrix(
            config_dict['data']['adj_filename'], 
            self.num_of_vertices,
            config_dict['data'].get('id_filename', None)
        )
        
        # 构建模型
        self.model = make_model(
            self.device, 
            self.nb_block, 
            self.in_channels,
            self.K, 
            self.nb_chev_filter,
            self.nb_time_filter,
            self.time_strides,
            self.adj_mx,
            self.num_for_predict,
            self.len_input,
            self.num_of_vertices
        )
        
        # 加载权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ 模型加载成功: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model.eval()
        
    def preprocess(self, input_data, mean, std):
        """
        数据预处理 - 标准化
        
        Args:
            input_data: numpy array, shape (B, N, F, T) 或 (N, F, T)
            mean: 均值
            std: 标准差
            
        Returns:
            torch tensor, shape (B, N, F, T)
        """
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # 标准化
        normalized_data = (input_data - mean) / (std + 1e-8)
        
        # 转为 PyTorch tensor
        tensor_data = torch.from_numpy(normalized_data).type(torch.FloatTensor).to(self.device)
        
        return tensor_data
    
    def postprocess(self, output, mean, std):
        """
        数据后处理 - 反标准化
        
        Args:
            output: torch tensor, shape (B, N, T_out)
            mean: 均值
            std: 标准差
            
        Returns:
            numpy array, 反标准化后的预测结果
        """
        output_np = output.detach().cpu().numpy()
        
        # 反标准化
        denormalized = output_np * std + mean
        
        return denormalized
    
    def predict(self, input_data, mean, std):
        """
        执行单次预测
        
        Args:
            input_data: numpy array, shape (B, N, F, T) - 输入特征
            mean: 均值
            std: 标准差
            
        Returns:
            numpy array, shape (B, N, T_out) - 预测结果
        """
        with torch.no_grad():
            # 预处理
            tensor_data = self.preprocess(input_data, mean, std)
            
            # 模型推理
            output = self.model(tensor_data)
            
            # 后处理
            predictions = self.postprocess(output, mean, std)
        
        return predictions
    
    def predict_batch(self, batch_inputs, mean, std, batch_size=32):
        """
        批量预测（支持大数据量）
        
        Args:
            batch_inputs: list of numpy arrays 或 单个 numpy array
            mean: 均值
            std: 标准差
            batch_size: 批处理大小
            
        Returns:
            numpy array - 所有预测结果拼接
        """
        if isinstance(batch_inputs, np.ndarray):
            batch_inputs = [batch_inputs]
        
        all_predictions = []
        
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i+batch_size]
            batch_data = np.concatenate(batch, axis=0) if len(batch) > 1 else batch[0]
            
            predictions = self.predict(batch_data, mean, std)
            all_predictions.append(predictions)
        
        return np.concatenate(all_predictions, axis=0)
