# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
import json
import configparser
from typing import List, Optional

"""
是基于 FastAPI 构建的实时流式交通速度预测 API 服务，
核心特点是通过滚动缓冲区（buffer） 接收「单时间步」的交通节点数据（速度、时段特征等），累计足够的历史时间步后，
调用 ASTGCN 模型输出未来车速预测结果；同时支持前端轮询获取最新预测值，适配 “实时、流式” 的交通数据输入场景
"""

# Add parent directory to path to import lib and model
import sys
# todo 这里的路径是相对于 main.py 的，所以需要 '../app/backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ASTGCN_r import make_model
from lib.utils import get_adjacency_matrix

app = FastAPI(
    title="ASTGCN Real-time Traffic Speed Prediction",
    description="Backend API for real-time traffic speed prediction using ASTGCN",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global State ====================
state = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "net": None,
    "mean": None,
    "std": None,
    "buffer": None, # (N, F, T) - Rolling buffer for input
    "N": 307,       # Number of nodes
    "T_in": 12,     # Input time steps
    "F_in": 3,      # Input features (Speed, Hour, Day)
    "config": None,
    "latest_prediction": None # Store latest prediction for frontend polling
}

# ==================== Data Models ====================
class PredictionRequest(BaseModel):
    """
    预测请求模型
    用于接收交通速度预测请求的 JSON 数据模型。
    
    Attributes:
        values: 输入特征数据，2 维数组，shape: (N_nodes, F_features)
            - N_nodes: 节点数量
            - F_features: 特征维度（通常为 3，即 [Speed, HourNorm, DayNorm]）
        timestamp: 可选，请求的时间戳，用于记录请求发生的时间。
    """
    # Receives a single time step of data for all nodes
    # Shape: (N_nodes, F_features) 
    # F_features should be 3: [Speed, HourNorm, DayNorm]
    values: List[List[float]] 
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[List[float]] # Shape: (N_nodes, T_out)
    timestamp: Optional[str] = None

# ==================== Helper Functions ====================
def load_model(config_path, params_path):
    print(f"Loading model from {params_path} with config {config_path}")
    
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding='utf-8')
    
    data_cfg = cfg['Data']
    train_cfg = cfg['Training']
    
    adj_filename = data_cfg['adj_filename']
    num_of_vertices = int(data_cfg['num_of_vertices'])
    num_for_predict = int(data_cfg['num_for_predict'])
    len_input = int(data_cfg['len_input'])
    
    # Model parameters
    in_channels = int(train_cfg['in_channels']) # Should be 3 now
    nb_block = int(train_cfg['nb_block'])
    K = int(train_cfg['K'])
    nb_chev_filter = int(train_cfg['nb_chev_filter'])
    nb_time_filter = int(train_cfg['nb_time_filter'])
    time_strides = int(train_cfg['num_of_hours']) # Usually 1
    
    DEVICE = state["DEVICE"]
    
    # Load Adjacency Matrix
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, None)
    
    # Initialize Model
    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
            time_strides, adj_mx, num_for_predict, len_input, num_of_vertices)
    
    # Load Weights
    if os.path.exists(params_path):
        # Check if params is a directory or file
        if os.path.isdir(params_path):
            # Find the best model (assuming it ends with .pth)
            # For now, let's try to find a .pth file
            files = [f for f in os.listdir(params_path) if f.endswith('.pth')]
            if files:
                # Load the latest one or specific one? 
                # Let's assume there is one 'best_model.pth' or similar, or just pick the first
                # Usually we save as epoch_X.pth. 
                # Let's verify what we have later.
                # For safety, let's try to load 'masked_mae_best.pth' or similar if exists, else first one
                target_file = files[0]
                model_file = os.path.join(params_path, target_file)
                print(f"Loading weights from {model_file}")
                net.load_state_dict(torch.load(model_file, map_location=DEVICE))
            else:
                print(f"No .pth files found in {params_path}")
        else:
            net.load_state_dict(torch.load(params_path, map_location=DEVICE))
    else:
        print(f"Model path {params_path} does not exist!")
        return None

    net.eval()
    return net

def init_buffer():
    # Initialize buffer with zeros or load from historical data
    # Shape: (N, F, T)
    state["buffer"] = np.zeros((state["N"], state["F_in"], state["T_in"]), dtype=np.float32)

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    # Hardcoded paths for now, or load from env/config
    # Assuming we use PEMS04 config
    config_path = "configurations/PEMS04_astgcn.conf"
    
    # We need to find where the model weights are saved.
    # Usually in experiments/PEMS04/...
    # Let's try to find the latest experiment folder
    exp_base = "experiments/PEMS04"
    if os.path.exists(exp_base):
        subdirs = [os.path.join(exp_base, d) for d in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, d))]
        if subdirs:
            # Sort by modification time to get latest
            latest_exp = max(subdirs, key=os.path.getmtime)
            params_path = latest_exp
        else:
            params_path = ""
    else:
        params_path = ""

    print(f"Auto-detected model path: {params_path}")
    
    if params_path:
        state["net"] = load_model(config_path, params_path)
        
        # Load normalization params
        # We saved scaler_params.json in data/processed/
        import json
        scaler_path = "data/processed/scaler_params.json"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                params = json.load(f)
            # Make sure shapes are correct for broadcasting
            # params['mean'] is a list.
            # DataPreprocessor saved mean/std for all 3 channels (Flow, Occ, Speed).
            # Speed is channel 2.
            state["mean"] = params['mean'][2]
            state["std"] = params['std'][2]
            print(f"Loaded scaler params: mean={state['mean']}, std={state['std']}")
        else:
            print("Scaler params not found, using default")
            state["mean"] = 0
            state["std"] = 1
            
        init_buffer()

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    if state["net"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Update buffer
    # Input values shape: (N, F) -> (307, 3)
    new_data = np.array(req.values, dtype=np.float32) # (N, F)
    
    # Check dimensions
    if new_data.shape != (state["N"], state["F_in"]):
         raise HTTPException(status_code=400, detail=f"Input shape mismatch. Expected ({state['N']}, {state['F_in']}), got {new_data.shape}")
    
    # Roll buffer and append new data
    # Buffer: (N, F, T)
    # Shift left
    state["buffer"][:, :, :-1] = state["buffer"][:, :, 1:] # 左移一位
    # Assign new data to last time step
    state["buffer"][:, :, -1] = new_data # 新数据放入最后一位
    
    # Prepare tensor
    # Model expects (B, N, F, T)
    input_tensor = torch.from_numpy(state["buffer"]).unsqueeze(0).to(state["DEVICE"])
    
    # Inference
    with torch.no_grad():
        output = state["net"](input_tensor) # (B, N, T_out)
        
    # Post-process (Inverse Normalization)
    # Output is normalized speed
    prediction = output.cpu().numpy()[0] # (N, T_out)
    
    # Inverse Z-Score: x * std + mean
    prediction = prediction * state["std"] + state["mean"]
    
    # Store for frontend
    response = {
        "predictions": prediction.tolist(),
        "timestamp": req.timestamp
    }
    state["latest_prediction"] = response
    
    return response

@app.get("/latest_prediction", response_model=PredictionResponse)
def get_latest_prediction():
    if state["latest_prediction"] is None:
        raise HTTPException(status_code=404, detail="No prediction available yet")
    return state["latest_prediction"]

@app.get("/status")
def status():
    return {
        "status": "online" if state["net"] is not None else "offline",
        "device": str(state["DEVICE"]),
        "model_loaded": state["net"] is not None
    }
