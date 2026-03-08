# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.inference import ASTGCNInference
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError  # 1. 新增导入
from fastapi.responses import PlainTextResponse      # 2. 新增导入

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title="ASTGCN 实时车辆速度预测API",
    description="基于注意力时空图卷积网络的交通速度预测服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源。为了安全，生产环境应设为你的前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # 允许所有请求头
)

# 3. 添加自定义异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # 在控制台打印详细的错误信息
    logger.error(f"✗ 请求体验证失败: {exc}")
    # 返回一个纯文本的错误响应，方便调试
    return PlainTextResponse(str(exc), status_code=400)

# ==================== 数据模型 ====================
class PredictionRequest(BaseModel):
    """预测请求数据模型"""
    input_data: List[List[List[float]]]  # shape: (N_nodes, F_features, T_timestamps)
    node_ids: Optional[List[int]] = None  # 可选：节点 ID 列表
    timestamp: Optional[str] = None  # 可选：时间戳

class PredictionResponse(BaseModel):
    """预测响应数据模型"""
    status: str
    predictions: List[List[float]]  # shape: (N_nodes, T_output)
    confidence: float = None
    timestamp: Optional[str] = None


# ==================== 全局变量 ====================
inference_engine = None
config = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化推理引擎"""
    global inference_engine, config
    
    try:
        # 加载配置
        config_path = "config.json"
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            config = {
                "num_of_vertices": 307,
                "in_channels": 1,
                "nb_block": 2,
                "K": 3,
                "nb_chev_filter": 64,
                "nb_time_filter": 64,
                "num_for_predict": 12,
                "len_input": 12,
                "adj_filename": "data/adj_mx.npy",
                "id_filename": None
            }
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # 自动检测设备
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"✓ 使用设备: {device}")

        # 初始化推理引擎
        model_path = config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            logger.error(f"✗ 模型路径配置不正确或文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        inference_engine = ASTGCNInference(config, model_path, device=device)
        logger.info("✓ 推理引擎初始化成功")
        
    except Exception as e:
        logger.error(f"✗ 推理引擎初始化失败: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "ASTGCN Speed Prediction API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    执行预测
    
    Args:
        request: 预测请求数据
        
    Returns:
        预测结果和元信息
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未就绪")
    
    try:
        # 转换输入数据
        input_array = np.array(request.input_data, dtype=np.float32)  # (N, F, T)
            
        # 从配置中获取参数（支持嵌套和扁平两种结构）
        in_channels = config.get('in_channels', config.get('training', {}).get('in_channels', 1))
        len_input = config.get('len_input', config.get('data', {}).get('len_input', 12))
            
        # 验证输入形状
        if input_array.shape[1] != in_channels or input_array.shape[2] != len_input:
            raise ValueError(
                f"输入形状不匹配。期望：(*, {in_channels}, {len_input}), "
                f"实际：{input_array.shape}"
            )
            
        # 加载标准化参数（从训练数据）
        mean = np.array([[[0.0]]])  # 应从训练数据加载
        std = np.array([[[1.0]]])   # 应从训练数据加载
        
        # 执行预测
        predictions = inference_engine.predict(input_array, mean, std)
        
        # 格式化输出
        pred_list = predictions.squeeze().tolist()
        if isinstance(pred_list, (int, float)):
            pred_list = [[pred_list]]
        elif isinstance(pred_list[0], (int, float)):
            pred_list = [pred_list]
        
        return PredictionResponse(
            status="success",
            predictions=pred_list,
            confidence=0.95,
            timestamp=request.timestamp
        )
        
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """批量预测接口"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未就绪")
    
    try:
        batch_data = [np.array(req.input_data, dtype=np.float32) for req in requests]
        
        mean = np.array([[[0.0]]])
        std = np.array([[[1.0]]])
        
        predictions = inference_engine.predict_batch(batch_data, mean, std)
        
        return {
            "status": "success",
            "batch_size": len(requests),
            "predictions": predictions.tolist()
        }
        
    except Exception as e:
        logger.error(f"批量预测出错: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/config")
async def get_config():
    """获取当前模型配置"""
    if config is None:
        raise HTTPException(status_code=503, detail="配置未加载")
    
    # 从配置中获取参数（支持嵌套和扁平两种结构）
    in_channels = config.get('in_channels', config.get('training', {}).get('in_channels', 1))
    len_input = config.get('len_input', config.get('data', {}).get('len_input', 12))
    num_of_vertices = config.get('num_of_vertices', config.get('data', {}).get('num_of_vertices', 307))
    num_for_predict = config.get('num_for_predict', config.get('data', {}).get('num_for_predict', 12))
    
    return {
        "model_config": config,
        "input_shape": f"({num_of_vertices}, {in_channels}, {len_input})",
        "output_shape": f"({num_of_vertices}, {num_for_predict})"
    }


@app.get("/")
async def root():
    """API 首页"""
    return {
        "message": "欢迎使用 ASTGCN 实时车辆速度预测API",
        "endpoints": {
            "health": "/health - 健康检查",
            "predict": "POST /predict - 单次预测",
            "predict_batch": "POST /predict_batch - 批量预测",
            "config": "/config - 获取模型配置",
            "docs": "/docs - API文档"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
