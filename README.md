# ASTGCN: 实时车辆速度预测系统研究与实现

本项目基于 **Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)** 模型，旨在实现对高速公路交通流（特别是车辆速度）的实时预测。本项目已构建从原始数据预处理、时空特征提取、模型训练评估到结果可视化的完整流程。

---

## 🚀 项目整体管线 (Pipeline)

项目运行遵循以下四个核心阶段：

### 阶段 1：数据预处理与构图
*   **功能**：清洗原始 `.npz` 数据，处理缺失值，构建基于地理距离的图邻接矩阵，并生成多周期时空特征。
*   **核心脚本**：
    *   [DataPreprocessor.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/preprocessing/DataPreprocessor.py): 集成缺失值修复、Z-Score 归一化、时间嵌入、多周期采样及邻接矩阵构建。
*   **输出**：`data/processed/` 目录下的 `train_data.npz` (特征/标签)、`adj_mat.npy` (邻接矩阵) 及 `scaler_params.json` (归一化参数)。

### 阶段 2：数据适配与格式转换
*   **功能**：已集成至 DataPreprocessor，不再需要单独步骤。

### 阶段 3：模型训练与验证
*   **功能**：读取配置文件，加载时空特征数据与图结构，训练并保存最优模型。
*   **核心脚本**：
    *   [train_ASTGCN.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/training/train_ASTGCN.py): 主训练程序。
    *   [ASTGCN_r.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/models/ASTGCN_r.py): 模型架构实现（含时空注意力层）。
*   **配置**：[PEMS04_astgcn.conf](file:///d:/codefile/Python/project/graduation%20project/astgcn/configurations/PEMS04_astgcn.conf)。
*   **输出**：`experiments/PEMS04/` 下的 `.params` 权重文件及 TensorBoard 日志。

### 阶段 4：性能评估与结果导出
*   **功能**：在测试集上进行推理，计算 MAE/RMSE/MAPE 指标并保存预测结果。
*   **核心工具**：
    *   [utils.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/lib/utils.py) 中的 `predict_and_save_results_mstgcn` 函数。
*   **输出**：`output_epoch_X_test.npz`、`metrics_results_test.csv` 及 `predictions_compare_test.csv`。

---

## 🛠️ 如何启动 (Execution)

请确保在项目根目录下运行以下命令：

### 1. 预处理与生成数据 (必须)
```bash
python preprocessing/DataPreprocessor.py
```
此步骤将自动完成数据清洗、多周期采样（默认 0周1日1最近）并生成 `train_data.npz`。

### 2. 启动模型训练
```bash
python training/train_ASTGCN.py --config configurations/PEMS04_astgcn.conf
```

---

## 📂 目录结构说明

```text
astgcn/
├── app/                 # 后端 API 服务
├── configurations/      # 配置文件 (.conf)
├── data/                # 原始数据及 processed/ 结果
├── experiments/         # 训练产物
├── models/              # 模型核心代码
├── preprocessing/       # 数据预处理脚本
├── simulation/          # 数据流模拟器
├── training/            # 训练脚本
└── README.md            # 项目说明
```

---

## 📊 关键参数解读

*   **in_channels (5)**: 输入特征数（流量+占有率+速度+Hour+Day）。
*   **len_input (24)**: 输入时间步长（根据多周期采样自动计算，如 1日+1最近 = 12+12 = 24）。
*   **K (3)**: 切比雪夫多项式阶数。
*   **nb_block (2)**: 堆叠的时空块数量。


## 启动推理服务（最小示例）

- pip install fastapi uvicorn
- uvicorn api.server:app --reload
- POST /warmup 参数示例：
  - config_path=configurations/PEMS04_astgcn.conf
  - params_path=experiments/PEMS04/astgcn_r_h1d0w0_channel1_1.000000e-03/epoch_X.params
- POST /predict
  - JSON: {"values":[N个节点的最新速度]}
  - 返回 {"pred":[N×T_out]}（已反归一化）
- 实时模拟：
  - python scripts/realtime_simulator.py --base_graph_path ./data/PEMS04/PEMS04.npz --interval 0.1