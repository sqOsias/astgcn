# ASTGCN 实时交通速度预测系统 - 项目文档

## 1. 项目简介
本项目实现了一个基于 **ASTGCN (Attention Based Spatial-Temporal Graph Convolutional Networks)** 的实时车辆速度预测系统。系统包含完整的数据处理、模型训练、后端推理服务、实时数据模拟器和可视化前端。

## 2. 目录结构
```
astgcn/
├── app/                  # 后端应用模块
│   ├── main.py           # FastAPI 服务入口 (API 路由、模型加载、预测逻辑)
│   └── backend/          # 后端辅助代码 (如推理类封装，暂未完全迁移)
├── configurations/       # 配置文件目录
│   └── PEMS04_astgcn.conf # PEMS04 数据集的模型配置 (层数、卷积核、训练参数)
├── data/                 # 数据存放目录
│   ├── PEMS04/           # 原始数据集 (pemsd4.npz, pemsd4.csv)
│   └── processed/        # 预处理后数据 (train_data.npz, adj_mat.npy, scaler_params.json)
├── evaluate/             # 评估与分析脚本
│   ├── eda_comprehensive.py # EDA 探索性数据分析脚本 (生成图表)
│   └── verify_dataloader.py # DataLoader 验证脚本
├── frontend/             # 前端项目
│   └── public/
│       └── index.html    # 单页可视化大屏 (Vue.js + ECharts)
├── lib/                  # 公共工具库
│   ├── utils.py          # 数据加载、图矩阵构建、评估指标计算
│   └── metrics.py        # 评价指标函数 (MAE, RMSE, MAPE)
├── models/               # 模型定义
│   └── ASTGCN_r.py       # ASTGCN 模型核心代码 (注意力机制、时空卷积块)
├── preprocessing/        # 数据预处理模块
│   └── DataPreprocessor.py # 数据清洗、特征提取、归一化、数据集划分
├── scripts/              # 辅助 Shell 脚本 (可选)
├── simulation/           # 仿真模块
│   └── producer.py       # 实时数据流模拟器 (生产者)
└── training/             # 模型训练模块
    └── train_ASTGCN.py   # 模型训练主脚本
```

## 3. 核心模块详解

### 3.1 数据预处理 (`preprocessing/DataPreprocessor.py`)
*   **功能**: 将原始 PEMS 数据转换为模型可用的张量格式。
*   **核心逻辑**:
    1.  **加载数据**: 读取 `.npz` 文件中的流量/速度数据。
    2.  **构建邻接矩阵**: 根据 `.csv` 中的距离信息，使用高斯核函数计算节点间的语义相似度矩阵 (Adjacency Matrix)。
        *   *关键参数*: `sigma` (控制高斯核衰减速度)，`epsilon` (稀疏化阈值)。
    3.  **时空特征提取**:
        *   **Time Embedding**: 提取时间戳的 "Hour of Day" 和 "Day of Week" 特征并归一化，拼接到输入特征中。
    4.  **滑动窗口**: 将时间序列切分为 `(B, T, N, F)` 格式的样本 (默认窗口长度12)。
    5.  **归一化**: 对速度数据进行 Z-Score 标准化，并保存 `scaler_params.json` (均值/方差) 用于后续反归一化。
    6.  **保存**: 生成 `train_data.npz` (包含训练/验证/测试集) 和 `adj_mat.npy`。

### 3.2 模型定义 (`models/ASTGCN_r.py`)
*   **功能**: 定义 ASTGCN 网络结构。
*   **核心组件**:
    *   `Spatial_Attention_layer`: 空间注意力机制，动态捕捉节点间的相关性。
    *   `Temporal_Attention_layer`: 时间注意力机制，捕捉不同时间步的依赖关系。
    *   `ChebConv`: 切比雪夫图卷积，处理空间依赖。
    *   `ASTGCN_block`: 堆叠上述组件的时空卷积块。
    *   `make_model`: 模型构建工厂函数。

### 3.3 模型训练 (`training/train_ASTGCN.py`)
*   **功能**: 执行模型的训练、验证和测试。
*   **流程**:
    1.  **加载配置**: 读取 `.conf` 文件中的超参数 (Batch Size, Epochs, Learning Rate)。
    2.  **加载数据**: 使用 `lib.utils.load_graphdata_channel1` 读取预处理后的数据。
    3.  **初始化模型**: 构建 ASTGCN 实例，定义 Loss 函数 (Masked MAE/MSE) 和优化器 (Adam)。
    4.  **训练循环**:
        *   前向传播 -> 计算 Loss -> 反向传播 -> 更新权重。
        *   每个 Epoch 结束后在验证集上评估，若 Loss 降低则保存模型 (`epoch_X.params`)。
    5.  **测试**: 选取验证集表现最好的模型在测试集上进行最终评估。

### 3.4 后端服务 (`app/main.py`)
*   **功能**: 提供实时预测 API。
*   **技术栈**: FastAPI, Uvicorn, PyTorch。
*   **核心接口**:
    *   `POST /predict`: 接收实时数据流 (Shape: `[N, F]`)，更新内存中的滑动窗口 Buffer，执行推理，返回预测结果。
    *   `GET /latest_prediction`: 供前端轮询，获取最新的预测数据。
    *   `GET /status`: 系统健康检查。
*   **特性**:
    *   启动时自动寻找并加载最新的模型权重文件。
    *   自动加载 `scaler_params.json` 进行反归一化，确保输出为真实速度值。

### 3.5 实时仿真 (`simulation/producer.py`)
*   **功能**: 模拟真实环境下的传感器数据推送。
*   **逻辑**:
    *   读取测试集数据 (`test_x`)。
    *   按照时间顺序，每隔一定间隔 (如 1秒) 提取一个时间步的数据。
    *   发送 HTTP POST 请求到后端 `/predict` 接口。

### 3.6 前端可视化 (`frontend/public/index.html`)
*   **功能**: 实时展示路网状态。
*   **技术栈**: Vue.js 3, ECharts, Tailwind CSS。
*   **组件**:
    *   **折线图**: 展示特定节点的实时速度预测曲线。
    *   **热力图**: 展示全路网 307 个节点的拥堵分布 (颜色编码)。
    *   **日志面板**: 显示数据接收状态。

## 4. 运行指南

### 步骤 1: 启动后端服务
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 步骤 2: 启动数据模拟器
```bash
python simulation/producer.py --interval 1.0
```

### 步骤 3: 访问前端
直接在浏览器打开 `frontend/public/index.html` (或通过 HTTP 服务托管)。

## 5. 待办事项 / 优化方向
1.  **模型训练**: 当前使用的是未充分训练的模型，需运行 `training/train_ASTGCN.py` 获取高精度权重。
2.  **前端优化**: 热力图目前是 1D 条形，若有节点经纬度信息，可升级为地图可视化 (Leaflet/Mapbox)。
3.  **部署**: 容器化 (Docker) 部署整个微服务架构。
