# 项目优化说明

## 📋 优化内容总览

本次优化在保留原有功能的基础上，增加了以下功能：

1. ✅ **增强的模型训练** (`3_train_model.py`)
   - 特征标准化（可选）
   - 交叉验证（5折）
   - 更全面的评估指标（R²、RMSE、MAE、MAPE）

2. ✅ **多模型对比** (`3.5_model_comparison.py`) - **新增**
   - OLS vs Ridge vs Lasso vs ElasticNet
   - 自动网格搜索最优超参数
   - 自动选择最佳模型并保存

3. ✅ **可视化分析** (`4_visualization.py`) - **新增**
   - 预测值 vs 实际值散点图
   - 残差分析图（残差图 + Q-Q图）
   - 特征重要性图
   - 残差分布直方图

4. ✅ **详细模型评估** (`5_model_evaluation.py`) - **新增**
   - 残差正态性检验
   - 异方差检验
   - 特征重要性分析
   - 生成评估报告（JSON格式）

5. ✅ **配置增强** (`config.py`)
   - 新增模型配置选项
   - 新增输出路径配置

## 🚀 使用流程

### 基础流程（仅OLS模型）

```bash
# Step 1: 数据预处理
python 1_preprocess.py

# Step 2: VIF共线性检查（手动查看，决定是否剔除特征）
python 2_check_vif.py

# Step 3: 训练OLS模型（增强版）
python 3_train_model.py

# Step 4: 生成可视化图表
python 4_visualization.py

# Step 5: 详细模型评估
python 5_model_evaluation.py
```

### 推荐流程（多模型对比）

```bash
# Step 1: 数据预处理
python 1_preprocess.py

# Step 2: VIF共线性检查（手动查看，决定是否剔除特征）
python 2_check_vif.py

# Step 3.5: 多模型对比（推荐！会自动选择最佳模型）
python 3.5_model_comparison.py

# Step 4: 生成可视化图表
python 4_visualization.py

# Step 5: 详细模型评估
python 5_model_evaluation.py
```

## 📁 输出文件说明

### 模型文件
- `output/ols_model.pkl` - OLS模型（如果运行3_train_model.py）
- `output/ols_model_ridge.pkl` - Ridge模型（如果是最佳模型）
- `output/ols_model_lasso.pkl` - Lasso模型（如果是最佳模型）
- `output/ols_model_elasticnet.pkl` - ElasticNet模型（如果是最佳模型）
- `output/ols_model_*_scaler.pkl` - 标准化器（如果使用了标准化）

### 对比结果
- `output/ols_model_comparison.csv` - 模型对比结果表格

### 可视化图表
- `output/plots/predicted_vs_actual.png` - 预测值vs实际值
- `output/plots/residual_analysis.png` - 残差分析
- `output/plots/feature_importance.png` - 特征重要性
- `output/plots/residual_distribution.png` - 残差分布

### 评估报告
- `output/evaluation_report.json` - 详细评估报告（JSON格式）

## ⚙️ 配置说明

在 `config.py` 中可以调整以下参数：

```python
USE_FEATURE_SCALING = True   # 是否使用特征标准化（正则化模型建议开启）
USE_CROSS_VALIDATION = True  # 是否使用交叉验证
CV_FOLDS = 5                 # 交叉验证折数
RANDOM_STATE = 42            # 随机种子
```

## 🔍 各模块功能详解

### 1. `3_train_model.py` (增强版)

**新增功能：**
- 特征标准化（可选，通过config控制）
- 5折交叉验证（可选，通过config控制）
- 更全面的评估指标：R²、RMSE、MAE、MAPE

**输出：**
- 模型文件（.pkl）
- 标准化器（如果使用标准化）
- 控制台输出详细统计报告

### 2. `3.5_model_comparison.py` (新增)

**功能：**
- 自动训练4种模型：OLS、Ridge、Lasso、ElasticNet
- 使用网格搜索自动调优超参数
- 对比所有模型的性能
- 自动保存最佳模型

**输出：**
- 最佳模型文件
- 对比结果CSV文件
- 控制台输出对比表格

### 3. `4_visualization.py` (新增)

**功能：**
- 生成4种可视化图表
- 自动识别模型类型（statsmodels或sklearn）
- 自动处理标准化（如果模型使用了标准化）

**输出：**
- 4个PNG图表文件（保存在output/plots/）

### 4. `5_model_evaluation.py` (新增)

**功能：**
- 残差正态性检验（Shapiro-Wilk或Kolmogorov-Smirnov）
- 异方差检验（Breusch-Pagan，仅statsmodels模型）
- 特征重要性分析
- 生成JSON格式评估报告

**输出：**
- 控制台输出详细统计检验结果
- JSON格式评估报告

## 💡 使用建议

1. **首次运行**：建议使用多模型对比流程（`3.5_model_comparison.py`），找到最佳模型
2. **特征选择**：运行`2_check_vif.py`后，手动查看VIF结果，在`config.py`中手动剔除高共线性特征
3. **模型解释**：如果注重模型可解释性，使用OLS模型；如果注重预测性能，使用正则化模型
4. **可视化**：运行`4_visualization.py`生成图表，用于报告和展示
5. **评估报告**：运行`5_model_evaluation.py`生成详细的统计检验报告

## ⚠️ 注意事项

1. **特征标准化**：正则化模型（Ridge/Lasso/ElasticNet）建议开启标准化，OLS模型可选
2. **模型兼容性**：可视化模块和评估模块会自动识别模型类型（statsmodels或sklearn）
3. **文件依赖**：确保按顺序运行，每个步骤的输出是下一步的输入
4. **手动特征剔除**：VIF检查后，如需剔除特征，请在`config.py`的`NUMERIC_COLS`或`BINARY_COLS`中手动删除

## 📊 评估指标说明

- **R² (决定系数)**：越接近1越好，表示模型解释的方差比例
- **RMSE (均方根误差)**：越小越好，表示预测误差的大小
- **MAE (平均绝对误差)**：越小越好，表示平均预测误差
- **MAPE (平均绝对百分比误差)**：越小越好，表示相对误差百分比

## 🔧 故障排除

1. **模型加载失败**：确保先运行模型训练脚本
2. **标准化器缺失**：如果模型使用了标准化，确保scaler文件存在
3. **图表中文乱码**：系统会自动尝试多种中文字体，如仍有问题请检查系统字体
4. **内存不足**：如果数据量很大，可以关闭交叉验证或减少CV折数

