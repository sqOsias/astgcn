# ASTGCN: 实时车辆速度预测系统研究与实现

本项目基于 **Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)** 模型，旨在实现对高速公路交通流（特别是车辆速度）的实时预测。本项目已构建从原始数据预处理、时空特征提取、模型训练评估到结果可视化的完整流程。

---

## 🚀 项目整体管线 (Pipeline)

项目运行遵循以下四个核心阶段：

### 阶段 1：数据预处理与构图
*   **功能**：清洗原始 `.npz` 数据，处理缺失值，构建基于地理距离的图邻接矩阵。
*   **核心脚本**：
    *   [DataPreprocessor.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/preProcessing/DataPreprocessor.py): 包含缺失值修复（线性插值）、Z-Score 归一化及高斯核邻接矩阵构建逻辑。
    *   [visualize.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/preProcessing/visualize.py): 生成预处理对比图、时空热力图及邻接矩阵热力图。
*   **输出**：`data/processed/` 目录下的 `train_data.npz` (特征/标签)、`adj_mat.npy` (邻接矩阵) 及 `scaler_params.pkl` (归一化参数)。

### 阶段 2：数据适配与格式转换
*   **功能**：将处理后的数据适配为 ASTGCN 官方训练脚本所需的特定 `.npz` 格式。
*   **核心脚本**：
    *   [convert_processed_to_astgcn.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/preProcessing/convert_processed_to_astgcn.py): 确保输入通道为速度（索引2），并对齐张量维度。
*   **输出**：`data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz` (按 r/d/w 约定的数据包)。

### 阶段 3：模型训练与验证
*   **功能**：读取配置文件，加载时空特征数据与图结构，训练并保存最优模型。
*   **核心脚本**：
    *   [train_ASTGCN_r.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/train_ASTGCN_r.py): 主训练程序。
    *   [ASTGCN_r.py](file:///d:/codefile/Python/project/graduation%20project/astgcn/model/ASTGCN_r.py): 模型架构实现（含时空注意力层）。
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

### 1. 预处理与可视化 (可选)
```bash
python preProcessing/DataPreprocessor.py
python preProcessing/visualize.py
```

### 2. 数据格式转换 (必须)
将清洗后的数据适配给 ASTGCN 训练引擎：
```bash
python preProcessing/convert_processed_to_astgcn.py ^
  --processed_dir "./data/processed" ^
  --base_graph_path "./data/PEMS04/PEMS04.npz" ^
  --num_of_hours 1 --num_of_days 0 --num_of_weeks 0 ^
  --input_feature_index 2
```

### 3. 启动模型训练
```bash
python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf
```

---

## 📂 目录结构说明

```text
astgcn/
├── configurations/      # 配置文件 (.conf)，定义数据集路径、模型超参、训练策略
├── data/                # 原始数据 (.npz, .csv) 及处理后的中间数据
├── experiments/         # 训练产物：最优模型权重 (.params)、推理结果 (.csv, .npz)
├── fig/                 # 可视化图像输出目录
├── lib/                 # 通用工具函数 (数据加载、指标计算、拉普拉斯计算)
├── model/               # 模型核心代码 (ASTGCN, MSTGCN)
├── preProcessing/       # 数据清洗、转换及可视化脚本
├── train_ASTGCN_r.py    # 训练入口脚本
└── prepareData.py       # 官方基础数据切分脚本
```

---

## 📊 关键参数解读

*   **r/d/w (1/0/0)**: 时间窗口尺度。当前配置仅使用最近 1 小时 (`r=1`) 的历史数据。
*   **in_channels (1)**: 输入特征数。已通过转换脚本锁定为速度通道。
*   **K (3)**: 切比雪夫多项式阶数，决定了图卷积捕捉空间邻域的深度。
*   **nb_block (2)**: 堆叠的时空块数量。
*   **loss_function (mse)**: 训练损失函数。建议针对缺失值较多的数据集尝试 `masked_mae`。
