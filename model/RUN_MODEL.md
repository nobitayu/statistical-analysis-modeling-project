**模型运行说明（model/）**

说明：本目录包含与模型训练、阈值优化、结果对比相关的脚本与输出。

文件列表（主要）：
- `train_trend_model.py`：训练并保存逻辑回归 pipeline（输出 `model_pipeline.joblib`、混淆矩阵 `confusion_matrix.png`、ROC 曲线 `roc_curve.png`、示例预测 `example_predictions.csv`）。
- `optimize_model.py`：对已保存模型进行阈值搜索与评估，生成 `optimize_results/` 下的图表与报告（threshold_metrics CSV、precision-recall 图、metrics-vs-threshold 图、confusion matrix 图、report JSON）。
- `compare_results.py`：从 `optimize_results/` 中读取最新阈值表，比较阈值 0.5 与优化后阈值并把对比追加到 `MODEL_REPORT_COMPLETE.md`。
- `model_pipeline.joblib`：已保存的模型 pipeline（二进制）。
- `optimize_results/`：阈值搜索结果和图表目录。

运行前准备（推荐使用虚拟环境）

1. 创建并激活虚拟环境（Windows PowerShell）：
```powershell
py -3.13 -m venv .venv313_sk18
.\.venv313_sk18\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
如果没有 `requirements.txt`，请安装至少以下包：
```powershell
pip install scikit-learn==1.8.0 joblib pandas matplotlib numpy
```

2. 工作目录说明：脚本使用相对路径（基于本 `model/` 目录），无需切换到 `process/`：
- `train_trend_model.py` 会从 `../process/youtube_data_balanced_5000.csv` 读取数据并在 `model/` 下保存模型与图像文件。
- `optimize_model.py` 默认读取 `../process/youtube_data_balanced_5000.csv`，加载 `model_pipeline.joblib`（位于 `model/`），并把结果写入 `model/optimize_results/`。
- `compare_results.py` 默认读取 `model/optimize_results/` 中最新的阈值 CSV 并把比较追加到 `model/MODEL_REPORT_COMPLETE.md`。

常用命令

- 训练并保存模型：
```powershell
py -3.13 model/train_trend_model.py
```
（或在已激活的 venv 中用 `python model/train_trend_model.py`）

- 运行阈值优化：
```powershell
py -3.13 model/optimize_model.py --model model/model_pipeline.joblib --data ../process/youtube_data_balanced_5000.csv
```
- 追加对比到报告：
```powershell
py -3.13 model/compare_results.py
```

生成的主要文件

- `model/model_pipeline.joblib`：训练好的 pipeline。
- `model/confusion_matrix.png`、`model/roc_curve.png`：训练时的评估图。
- `model/example_predictions.csv`：示例预测输出。
- `model/optimize_results/threshold_metrics_YYYYMMDD_HHMMSS.csv`：每个阈值对应的指标表。
- `model/optimize_results/precision_recall_YYYYMMDD_HHMMSS.png`。
- `model/optimize_results/metrics_vs_threshold_YYYYMMDD_HHMMSS.png`。
- `model/optimize_results/confusion_matrix_YYYYMMDD_HHMMSS.png`。
- `model/optimize_results/report_YYYYMMDD_HHMMSS.json`。
- `model/MODEL_REPORT_COMPLETE.md`：完整的模型报告（含优化前后对比）。

路径变更说明

- 我已将 `train_trend_model.py`、`optimize_model.py` 中的默认路径修改为基于脚本目录（`__file__`）的相对路径，确保在 `model/` 目录下运行时能正确定位数据（在 `process/`）和输出到 `model/`。
- `compare_results.py` 使用 `model/optimize_results/` 作为默认输出目录，无需修改。

需要我帮你做的后续事项（可选）
- 生成 `requirements.txt`（根据虚拟环境导出）；
- 把 `model/` 下脚本改为库函数形式（便于在 notebook 中调用）；
- 运行一次完整流程并把生成的关键文件列出来。

如需我现在做其中某项，请回复要执行的操作。