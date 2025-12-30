# -*- coding: utf-8 -*-
"""
训练与保存预测未发布视频是否会热门的逻辑回归模型
Usage:
    python train_trend_model.py

功能:
- 读取已处理并平衡的数据: youtube_data_balanced_5000.csv
- 构建预处理 pipeline (数值、类别、文本 TF-IDF + SVD)
- 训练 LogisticRegression 模型 (GridSearchCV 调参)
- 评估并保存模型 pipeline (joblib)
- 提供 predict_unpublished 接口
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

RND = 42

SCRIPT_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'process', 'youtube_data_balanced_5000.csv')
MODEL_OUT = os.path.join(SCRIPT_DIR, 'model_pipeline.joblib')

# 需要用于未发布视频预测的特征（不要使用发布后才有的 view/likes/comment）
NUMERIC_FEATURES = ['title_length', 'question_exclamation_count', 'tag_count', 'hour', 'tags_in_title', 'is_weekend']
CATEGORICAL_FEATURES = ['time_period', 'category']
TEXT_FEATURE = 'title'
TARGET = 'is_trending'


def load_data(path=DATA_FILE):
    print(f"读取数据: {path}")
    df = pd.read_csv(path)
    # 简单检查
    missing_target = df[TARGET].isna().sum()
    if missing_target > 0:
        raise ValueError(f"目标列 {TARGET} 存在缺失: {missing_target} 行")
    # 确保数值列为数值类型
    for c in ['title_length', 'question_exclamation_count', 'tag_count', 'hour']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def build_pipeline():
    # 数值管道
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 文本处理(标题): TF-IDF -> SVD 降维
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')),
        ('svd', TruncatedSVD(n_components=50, random_state=RND))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ('title', text_pipeline, TEXT_FEATURE)
    ], remainder='drop')

    pipe = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(solver='saga', max_iter=2000, random_state=RND))
    ])

    return pipe


def train_and_evaluate(df, save_model=True):
    # 准备 X, y（仅使用可用于未发布的特征）
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND, stratify=y)

    pipe = build_pipeline()

    # 使用 GridSearchCV 调参 C
    param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

    print("开始训练 (GridSearchCV)...")
    gs.fit(X_train, y_train)
    print(f"最佳参数: {gs.best_params_}")
    best_model = gs.best_estimator_

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n测试集评估报告:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.4f}")

    # 保存模型
    if save_model:
        joblib.dump(best_model, MODEL_OUT)
        print(f"模型已保存为: {MODEL_OUT}")

    # 绘图: 混淆矩阵与 ROC
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-trending', 'Trending'])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png', dpi=200, bbox_inches='tight')

    return best_model, (X_test, y_test, y_pred, y_prob)


def predict_unpublished(input_data, model_path=MODEL_OUT):
    """
    对未发布视频做预测。
    input_data: dict 或 pandas.DataFrame，包含需要的特征
    返回: DataFrame 包含原始输入、预测概率和预测标签
    """
    if isinstance(input_data, dict):
        X_new = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        X_new = input_data.copy()
    else:
        raise ValueError('input_data 必须是 dict 或 pandas.DataFrame')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model = joblib.load(model_path)

    probs = model.predict_proba(X_new)[:, 1]
    preds = model.predict(X_new)

    res = X_new.reset_index(drop=True).copy()
    res['pred_prob_trending'] = probs
    res['pred_is_trending'] = preds.astype(int)
    return res


def main():
    df = load_data()
    model, eval_data = train_and_evaluate(df)
    X_test, y_test, y_pred, y_prob = eval_data
    print('\n训练与评估完成。输出文件: model_pipeline.joblib, confusion_matrix.png, roc_curve.png')

    # 保存一个示例预测（从测试集随机取）
    example = X_test.sample(5, random_state=RND)
    example_res = predict_unpublished(example)
    example_res.to_csv('example_predictions.csv', index=False, encoding='utf-8-sig')
    print('示例预测已保存为 example_predictions.csv')


if __name__ == '__main__':
    main()
