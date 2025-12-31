# -*- coding: utf-8 -*-
"""
- 从已保存的模型加载
- 在测试数据上计算预测概率
- 在多个阈值上搜索最佳阈值
"""
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
    roc_curve,
)

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_pipeline.joblib')
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'process', 'youtube_data_balanced_5000.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'optimize_results')


def load_model(path: str):
    return joblib.load(path)


def get_probs(model, X: pd.DataFrame) -> np.ndarray:
    """返回正类的概率估计"""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return np.max(probs, axis=1)
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-df))
    preds = model.predict(X)
    return preds.astype(float)


def threshold_search(y_true: np.ndarray, probs: np.ndarray, thresholds: Optional[np.ndarray] = None):
    """在多个阈值上搜索最佳阈值"""
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
    """绘制Precision-Recall曲线"""
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
    """绘制指标随阈值变化的曲线"""
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


def plot_roc_curve(probs: np.ndarray, y_true: np.ndarray, out_path: str):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """保存混淆矩阵图"""
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
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", 
                    color=("white" if cm[i, j] > thresh else "black"))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(args):
    """运行阈值优化"""
    model = load_model(args.model)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")

    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    probs = get_probs(model, X)

    # 计算AUC
    auc = None
    try:
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = None

    # 阈值搜索
    df_metrics = threshold_search(y.values, probs)

    # 选择最优阈值
    metric = args.metric.lower()
    if metric not in df_metrics.columns:
        metric = "f1"

    best_row = df_metrics.loc[df_metrics[metric].idxmax()]
    best_threshold = float(best_row["threshold"])

    y_pred_best = (probs >= best_threshold).astype(int)

    # 计算默认阈值0.5的指标
    y_pred_default = (probs >= 0.5).astype(int)
    default_metrics = {
        "threshold": 0.5,
        "precision": float(precision_score(y, y_pred_default, zero_division=0)),
        "recall": float(recall_score(y, y_pred_default, zero_division=0)),
        "f1": float(f1_score(y, y_pred_default, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred_default)),
    }

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存表格与图像
    df_metrics.to_csv(os.path.join(out_dir, f"threshold_metrics_{timestamp}.csv"), index=False)
    plot_precision_recall(probs, y.values, os.path.join(out_dir, f"precision_recall_{timestamp}.png"))
    plot_metric_vs_threshold(df_metrics, os.path.join(out_dir, f"metrics_vs_threshold_{timestamp}.png"))
    plot_roc_curve(probs, y.values, os.path.join(out_dir, f"roc_curve_{timestamp}.png"))
    save_confusion_matrix(y.values, y_pred_best, os.path.join(out_dir, f"confusion_matrix_{timestamp}.png"))

    # 生成报告
    report = {
        "model": args.model,
        "data": args.data,
        "target": args.target,
        "auc": auc,
        "default_threshold_metrics": default_metrics,
        "best_metric": metric,
        "best_threshold": best_threshold,
        "best_metric_value": float(best_row[metric]),
        "best_threshold_metrics": {
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "accuracy": float(best_row["accuracy"]),
        }
    }

    # 保存带阈值的模型副本
    saved_model_path = os.path.join(out_dir, f"model_with_threshold_{timestamp}.joblib")
    try:
        joblib.dump({"pipeline": model, "threshold": best_threshold}, saved_model_path)
        report["saved_model"] = saved_model_path
    except Exception:
        report["saved_model"] = None

    with open(os.path.join(out_dir, f"report_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("阈值优化完成")
    print("="*80)
    print(f"\nAUC值: {auc:.4f}")
    print(f"\n默认阈值 (0.5) 的指标:")
    print(f"  Precision: {default_metrics['precision']:.4f}")
    print(f"  Recall:    {default_metrics['recall']:.4f}")
    print(f"  F1 Score:  {default_metrics['f1']:.4f}")
    print(f"  Accuracy:  {default_metrics['accuracy']:.4f}")
    
    print(f"\n最优阈值 ({best_threshold:.3f}, 基于 {metric}) 的指标:")
    print(f"  Precision: {report['best_threshold_metrics']['precision']:.4f}")
    print(f"  Recall:    {report['best_threshold_metrics']['recall']:.4f}")
    print(f"  F1 Score:  {report['best_threshold_metrics']['f1']:.4f}")
    print(f"  Accuracy:  {report['best_threshold_metrics']['accuracy']:.4f}")
    
    print(f"\n性能提升:")
    print(f"  Precision: {report['best_threshold_metrics']['precision'] - default_metrics['precision']:+.4f}")
    print(f"  Recall:    {report['best_threshold_metrics']['recall'] - default_metrics['recall']:+.4f}")
    print(f"  F1 Score:  {report['best_threshold_metrics']['f1'] - default_metrics['f1']:+.4f}")
    print(f"  Accuracy:  {report['best_threshold_metrics']['accuracy'] - default_metrics['accuracy']:+.4f}")
    
    print(f"\n结果已保存到: {out_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description="Optimize classification threshold for model")
    p.add_argument("--model", default=MODEL_PATH, help="path to saved model pipeline (joblib)")
    p.add_argument("--data", default=DATA_FILE, help="path to CSV with features + target")
    p.add_argument("--target", default="is_trending", help="name of target column in CSV")
    p.add_argument("--metric", default="f1", help="metric to optimize: f1, youden, precision, recall, balanced_accuracy")
    p.add_argument("--output-dir", dest="output_dir", default=OUTPUT_DIR, help="directory to save outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)

