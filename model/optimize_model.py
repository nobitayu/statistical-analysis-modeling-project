"""模型阈值优化与评估脚本

用法示例:
python optimize_model.py --model process/model_pipeline.joblib --data process/youtube_data_balanced_5000.csv

脚本功能：
- 从已保存的 pipeline 中加载模型
- 在给定测试数据上计算预测概率
- 在多个阈值上搜索最佳阈值（基于 F1、Youden 等指标）
- 生成并保存 PR 曲线、F1-vs-threshold 曲线、混淆矩阵图和报告（JSON）
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_model(path: str):
    return joblib.load(path)


def get_probs(model, X: pd.DataFrame) -> np.ndarray:
    """返回正类的概率估计，必要时从 decision_function 转换。"""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # 二分类取正类概率
        if probs.shape[1] == 2:
            return probs[:, 1]
        # 若多分类，返回对每行最大类的概率（不常见）
        return np.max(probs, axis=1)

    if hasattr(model, "decision_function"):
        # 将决策函数映射到 (0,1) via sigmoid
        df = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-df))

    # 最后尝试 predict（退化）
    preds = model.predict(X)
    return preds.astype(float)


def threshold_search(y_true: np.ndarray, probs: np.ndarray, thresholds: Optional[np.ndarray] = None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        # Youden index = TPR - FPR = recall - (1 - specificity)
        # specificity = TN/(TN+FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = r - (1 - specificity)

        rows.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "youden": float(youden),
        })

    return pd.DataFrame(rows)


def plot_precision_recall(probs: np.ndarray, y_true: np.ndarray, out_path: str):
    precision, recall, thresh = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_vs_threshold(df_metrics: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics["threshold"], df_metrics["precision"], label="Precision")
    plt.plot(df_metrics["threshold"], df_metrics["recall"], label="Recall")
    plt.plot(df_metrics["threshold"], df_metrics["f1"], label="F1")
    plt.plot(df_metrics["threshold"], df_metrics["youden"], label="Youden")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["not_trend", "trend"])
    plt.yticks(tick_marks, ["not_trend", "trend"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=("white" if cm[i, j] > thresh else "black"))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(args):
    model = load_model(args.model)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")

    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    probs = get_probs(model, X)

    # 基本统计
    auc = None
    try:
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = None

    df_metrics = threshold_search(y.values, probs)

    # 选择最优阈值
    metric = args.metric.lower()
    if metric not in df_metrics.columns:
        metric = "f1"

    best_row = df_metrics.loc[df_metrics[metric].idxmax()]
    best_threshold = float(best_row["threshold"])

    y_pred_best = (probs >= best_threshold).astype(int)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 保存表格与图像
    df_metrics.to_csv(os.path.join(out_dir, f"threshold_metrics_{timestamp}.csv"), index=False)
    plot_precision_recall(probs, y.values, os.path.join(out_dir, f"precision_recall_{timestamp}.png"))
    plot_metric_vs_threshold(df_metrics, os.path.join(out_dir, f"metrics_vs_threshold_{timestamp}.png"))
    save_confusion_matrix(y.values, y_pred_best, os.path.join(out_dir, f"confusion_matrix_{timestamp}.png"))

    report = {
        "model": args.model,
        "data": args.data,
        "target": args.target,
        "auc": auc,
        "best_metric": metric,
        "best_threshold": best_threshold,
        "best_metric_value": float(best_row[metric]),
    }

    with open(os.path.join(out_dir, f"report_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Optimization finished")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description="Optimize classification threshold and produce evaluation plots/reports")
    script_dir = os.path.dirname(__file__)
    default_model = os.path.join(script_dir, 'model_pipeline.joblib')
    default_data = os.path.join(script_dir, '..', 'process', 'youtube_data_balanced_5000.csv')
    default_out = os.path.join(script_dir, 'optimize_results')
    p.add_argument("--model", default=default_model, help="path to saved model pipeline (joblib)")
    p.add_argument("--data", default=default_data, help="path to CSV with features + target")
    p.add_argument("--target", default="is_trending", help="name of target column in CSV")
    p.add_argument("--metric", default="f1", help="metric to optimize: f1, youden, precision, recall, balanced_accuracy")
    p.add_argument("--output-dir", dest="output_dir", default=default_out, help="directory to save outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
