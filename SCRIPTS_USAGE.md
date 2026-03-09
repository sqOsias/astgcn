# 脚本使用指南 - 快速参考

本文件提供所有脚本的快速使用方法和命令示例。



## 🚀 快速开始（推荐流程）

### Step 1: 数据准备
```bash
# 预处理原始数据
python preProcessing/DataPreprocessor.py
python preProcessing/convert_processed_to_astgcn.py
```
#### 输出说明 
控制台会显示：
 - 原始数据缺失率统计（通常 8-12%）
 - 数据插值修复完成情况
 - 归一化均值和标准差
 - 邻接矩阵的稀疏度（非零元素比例约 20%）
 - 最终生成的特征张量形状，例如：(10000, 307, 3, 12)
#### 数据格式转换产生的结果文件：
data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz - ASTGCN 专用格式

该文件包含：
 - train_x, train_target - 训练集特征和标签
 - val_x, val_target - 验证集特征和标签
 - test_x, test_target - 测试集特征和标签
 - mean, std - 归一化参数
### Step 2: 数据分析（可选但推荐）
```bash
# 分析数据质量和处理效果
python analyze_data_quality.py
# 输出：fig/analysis/ 下的可视化图表和报告
```

### Step 3: 模型训练
```bash
# GPU 训练（推荐）
python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf

# CPU 训练（无 GPU 时）
python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf --cpu
```

### Step 4: 模型评估
```bash
# 评估最佳模型
python evaluate_model.py \
  --config configurations/PEMS04_astgcn.conf \
  --model_path experiments/PEMS04/astgcn_h1d0w0_channel1_0.001000/epoch_26.params
```

---

## 📝 详细使用说明

### 1️⃣ 完整训练流程脚本

**文件名**: `run_train_pipeline.py`

**功能特点**:
- ✅ 自动化的端到端训练
- ✅ 实时进度显示
- ✅ 自动生成训练报告
- ✅ 支持断点续训
- ✅ CPU/GPU 自动检测

**基本用法**:
```bash
# GPU 训练
python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf

# CPU 训练
python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf --cpu

# 使用不同配置文件
python run_train_pipeline.py --config configurations/PEMS08_astgcn.conf
```

**输出文件**:
```
experiments/PEMS04/astgcn_*/
├── epoch_X.params              # 模型权重
├── metrics_results_test.csv    # 评估指标
├── output_epoch_X_test.npz    # 预测结果
├── predictions_compare_test.csv
└── training_summary.txt       # 训练报告
```

**预计运行时间**:
- GPU: 2-4 小时（完整 80 epochs）
- CPU: 8-12 小时（完整 80 epochs）

---

### 2️⃣ 数据质量分析脚本

**文件名**: `analyze_data_quality.py`

**功能特点**:
- ✅ 缺失值统计分析
- ✅ 数据分布可视化
- ✅ 处理前后对比
- ✅ 自动生成报告

**基本用法**:
```bash
python analyze_data_quality.py
```

**前提条件**:
- 已运行 `DataPreprocessor.py`
- `data/processed/` 目录下有处理后的数据

**输出文件**:
```
fig/analysis/
├── raw_data_timeseries.png      # 原始数据时序图
├── before_after_comparison.png  # 处理前后对比
├── adjacency_matrix_heatmap.png # 邻接矩阵热力图
└── analysis_report.txt          # 分析报告
```

**预计运行时间**: 5-10 分钟

**典型输出示例**:
```
============================================================
缺失值分析
============================================================

原始数据:
  - 总元素数：1,663,488
  - 0 值数量：166,349
  - 缺失率：10.00%
  
  - 流量：缺失率 = 8.50%
  - 占有率：缺失率 = 12.30%
  - 速度：缺失率 = 9.20%
```

---

### 3️⃣ 模型评估与验证脚本

**文件名**: `evaluate_model.py`

**功能特点**:
- ✅ 多维度评估指标计算
- ✅ 丰富的可视化图表
- ✅ 支持 CPU/GPU
- ✅ 自动生成 CSV 报告

**基本用法**:
```bash
# GPU 评估
python evaluate_model.py \
  --config configurations/PEMS04_astgcn.conf \
  --model_path experiments/PEMS04/astgcn_*/epoch_26.params

# CPU 评估
python evaluate_model.py \
  --config configurations/PEMS04_astgcn.conf \
  --model_path experiments/PEMS04/astgcn_*/epoch_26.params \
  --cpu

# 自定义输出目录
python evaluate_model.py \
  --config configurations/PEMS04_astgcn.conf \
  --model_path experiments/PEMS04/astgcn_*/epoch_26.params \
  --output_dir my_evaluation_results
```

**必需参数**:
- `--config`: 配置文件路径
- `--model_path`: 模型权重文件路径

**可选参数**:
- `--cpu`: 使用 CPU 进行评估
- `--output_dir`: 结果输出目录（默认：experiments/evaluation_results）

**输出文件**:
```
experiments/evaluation_results/
├── evaluation_report.csv         # CSV 格式评估报告
├── metrics_by_step.png           # 各步长指标趋势图
├── prediction_vs_truth_sample.png # 预测对比图
├── scatter_plot.png              # 散点图（含 R²）
└── error_distribution.png        # 误差分布图
```

**评估指标说明**:
```csv
Overall Metrics
Metric,Value
MAE,5.234
RMSE,7.891
MAPE (%),12.45

Per-Step Metrics
Step,MAE,RMSE,MAPE (%)
1,4.12,6.34,9.87
2,4.56,6.89,10.23
...
```

**预计运行时间**: 10-30 分钟（取决于测试集大小）

---

### 4️⃣ CPU 训练验证脚本

**文件名**: `test_cpu_training.py`

**功能特点**:
- ✅ 快速验证（2-5 分钟）
- ✅ 使用模拟数据
- ✅ 测试模型构建、训练、推理全流程

**基本用法**:
```bash
python test_cpu_training.py
```

**无需任何前提条件**，自动创建模拟数据进行测试。

**输出示例**:
```
============================================================
CPU 训练快速验证测试
============================================================

✓ 使用设备：cpu

步骤 1: 创建模拟数据
✓ 节点数：50
✓ 输入形状：(batch, 50, 1, 3)
✓ 输出形状：(batch, 50, 3)

步骤 2: 构建 ASTGCN 模型
✓ 模型构建完成（用时：0.15s）
✓ 总参数量：1,234,567

步骤 3: 准备训练环境
✓ 损失函数：L1 Loss (MAE)
✓ 优化器：Adam (lr=0.001)

步骤 4: 执行模拟训练
开始训练（10 个 batch）...
  Batch 2/10: Loss = 0.9234
  Batch 4/10: Loss = 0.8765
  ...

✅ 所有测试通过！CPU 训练可行。
```

**预计运行时间**: 2-5 分钟

**适用场景**:
- 新环境配置验证
- 代码修改后快速测试
- 教学演示
- 无 GPU 时的可行性验证

---

### 5️⃣ 原始训练脚本（已增强）

**文件名**: `train_ASTGCN_r.py`

**功能特点**:
- ✅ 官方原始训练脚本
- ✅ 已添加 CPU 支持
- ✅ 适合需要精细控制的场景

**基本用法**:
```bash
# GPU 训练
python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf

# CPU 训练
python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf --cpu

# 指定 GPU 设备
CUDA_VISIBLE_DEVICES=1 python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf
```

**与 `run_train_pipeline.py` 的区别**:
- `train_ASTGCN_r.py`: 更接近底层，适合调试和定制
- `run_train_pipeline.py`: 更高级的封装，自动化程度更高

**推荐使用**: `run_train_pipeline.py`（除非你需要特殊定制）

---

## 🔧 常见使用场景

### 场景 1: 第一次运行项目

```bash
# 1. 检查数据
ls data/PEMS04/

# 2. 如果没有数据，下载并预处理
# （需要先获取 PEMS04 数据集）
python preProcessing/DataPreprocessor.py
python preProcessing/convert_processed_to_astgcn.py

# 3. 分析数据质量
python analyze_data_quality.py

# 4. 快速验证（可选）
python test_cpu_training.py

# 5. 开始训练（GPU）
python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf

# 6. 训练完成后评估
python evaluate_model.py \
  --config configurations/PEMS04_astgcn.conf \
  --model_path experiments/PEMS04/astgcn_*/epoch_*.params
```

---

### 场景 2: 只有 CPU 环境

```bash
# 1. 快速验证 CPU 可行性
python test_cpu_training.py

# 2. 使用简化模型训练（减少参数）
# 先修改配置文件 configurations/cpu_config.conf:
#   nb_block = 1
#   nb_chev_filter = 32
#   nb_time_filter = 32
#   batch_size = 8

python run_train_pipeline.py \
  --config configurations/cpu_config.conf \
  --cpu

# 3. CPU 评估
python evaluate_model.py \
  --config configurations/cpu_config.conf \
  --model_path experiments/PEMS04/astgcn_*/epoch_*.params \
  --cpu
```

---

### 场景 3: 调参实验

```bash
# 1. 创建多个配置文件
cp configurations/PEMS04_astgcn.conf configs/exp_lr0.001.conf
cp configurations/PEMS04_astgcn.conf configs/exp_lr0.0001.conf
cp configurations/PEMS04_astgcn.conf configs/exp_block3.conf

# 2. 修改各个文件的参数

# 3. 批量训练（后台运行）
nohup python run_train_pipeline.py --config configs/exp_lr0.001.conf &
nohup python run_train_pipeline.py --config configs/exp_lr0.0001.conf &
nohup python run_train_pipeline.py --config configs/exp_block3.conf &

# 4. 等待训练完成后分别评估
python evaluate_model.py --config configs/exp_lr0.001.conf --model_path ...
python evaluate_model.py --config configs/exp_lr0.0001.conf --model_path ...
python evaluate_model.py --config configs/exp_block3.conf --model_path ...

# 5. 对比结果
cat experiments/*/metrics_results_test.csv
```

---

### 场景 4: 断点续训

```bash
# 如果训练中断（例如 epoch 50 时）

# 方法 1: 自动续训（修改配置文件）
# 在 configurations/PEMS04_astgcn.conf 中设置:
#   start_epoch = 50

python run_train_pipeline.py --config configurations/PEMS04_astgcn.conf

# 方法 2: 直接指定 epoch（如果知道确切文件名）
python train_ASTGCN_r.py \
  --config configurations/PEMS04_astgcn.conf \
  --start_epoch 50
```

---

### 场景 5: 结果可视化与报告生成

```bash
# 1. 查看 TensorBoard 日志
tensorboard --logdir experiments/PEMS04/astgcn_*/runs/

# 2. 查看数据分析报告
cat fig/analysis/analysis_report.txt

# 3. 查看训练总结
cat experiments/PEMS04/astgcn_*/training_summary.txt

# 4. 查看评估报告
cat experiments/evaluation_results/evaluation_report.csv

# 5. 打开可视化图表（Linux/Mac）
eog fig/analysis/*.png
eog experiments/evaluation_results/*.png
```

---

## 📊 性能基准参考

### 训练时间对比

| 配置 | GPU | CPU | 备注 |
|------|-----|-----|------|
| 完整模型（80 epochs） | 2-4h | 8-12h | PEMS04 数据集 |
| 简化模型（80 epochs） | 1-2h | 4-6h | nb_block=1 |
| 快速验证（10 epochs） | 15-30min | 1-2h | 调试用 |

### 评估时间对比

| 操作 | GPU | CPU |
|------|-----|-----|
| 加载模型 | <1s | <1s |
| 测试集预测 | 2-5min | 10-20min |
| 指标计算 | <1min | <1min |
| 可视化生成 | 1-2min | 1-2min |

---

## 🐛 故障排查

### 问题 1: 找不到数据文件

```bash
# 错误信息：FileNotFoundError: data/PEMS04/PEMS04.npz

# 解决：
# 1. 确认数据已下载
ls data/PEMS04/

# 2. 如果缺失，运行预处理
python preProcessing/DataPreprocessor.py
```

### 问题 2: CUDA Out of Memory

```bash
# 错误信息：RuntimeError: CUDA out of memory

# 解决：
# 方法 1: 减小 batch_size（修改配置文件）
batch_size = 8  # 从 32 改为 8

# 方法 2: 使用 CPU
python run_train_pipeline.py --cpu

# 方法 3: 简化模型
nb_block = 1
nb_chev_filter = 32
```

### 问题 3: 模型加载失败

```bash
# 错误信息：FileNotFoundError: epoch_26.params

# 解决：
# 1. 确认训练已完成
ls experiments/PEMS04/astgcn_*/

# 2. 找到正确的 epoch 文件
# 查看 training_summary.txt 中的 best_epoch

# 3. 使用正确的路径
python evaluate_model.py \
  --model_path experiments/PEMS04/astgcn_*/epoch_26.params
```

---

## 📞 需要帮助？

1. **查看详细文档**: `TRAINING_GUIDE.md`
2. **检查日志**: 每个脚本都会生成详细的日志输出
3. **常见问题**: 参考本文档的故障排查部分
4. **实验记录**: 建议保存每次实验的配置和结果

---

## ✅ 检查清单

运行前的检查：

- [ ] Python >= 3.7
- [ ] 依赖包已安装 (`pip list`)
- [ ] 数据文件存在 (`ls data/PEMS04/`)
- [ ] 配置文件正确 (`cat configurations/PEMS04_astgcn.conf`)
- [ ] 有足够的磁盘空间（至少 5GB）
- [ ] 如果是 GPU 训练，确认 CUDA 可用 (`nvidia-smi`)

运行后的验证：

- [ ] 模型文件已生成 (`ls experiments/`)
- [ ] 评估报告已生成 (`cat experiments/*/metrics_results_test.csv`)
- [ ] 可视化图表正常显示
- [ ] 训练日志无 ERROR 级别错误

---

祝实验顺利！如有问题，请查看 `TRAINING_GUIDE.md` 获取详细说明。
