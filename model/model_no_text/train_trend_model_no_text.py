# -*- coding: utf-8 -*-
"""
训练与保存预测未发布视频是否会热门的逻辑回归模型（不使用文本特征版本）
Usage:
    python train_trend_model_no_text.py

功能:
- 读取已处理并平衡的数据: youtube_data_balanced_5000.csv
- 构建预处理 pipeline (仅使用数值和类别特征，不使用文本)
- 训练 LogisticRegression 模型 (GridSearchCV 调参)
- 评估并保存模型 pipeline (joblib)
- 提供 predict_unpublished 接口

与原始版本的区别:
- 完全不使用文本特征（title）
- 仅使用数值特征和类别特征
- 可以对比文本特征对模型性能的影响
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay, 
                             precision_score, recall_score, f1_score, accuracy_score)
import matplotlib.pyplot as plt

RND = 42

SCRIPT_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'process', 'youtube_data_balanced_5000.csv')
MODEL_OUT = os.path.join(SCRIPT_DIR, 'model_pipeline_no_text.joblib')

# 仅使用数值和类别特征，不使用文本
NUMERIC_FEATURES = ['title_length', 'question_exclamation_count', 'tag_count', 'hour', 'tags_in_title', 'is_weekend']
CATEGORICAL_FEATURES = ['time_period', 'category']
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

    # 类别特征处理
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ], remainder='drop')

    pipe = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(solver='saga', max_iter=2000, random_state=RND))
    ])

    return pipe


def train_and_evaluate(df, save_model=True):
    # 准备 X, y（仅使用数值和类别特征，不使用文本）
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND, stratify=y)

    pipe = build_pipeline()

    # 使用 GridSearchCV 调参 C
    param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

    print("开始训练 (GridSearchCV)...")
    print(f"使用特征: {NUMERIC_FEATURES + CATEGORICAL_FEATURES}")
    print(f"特征数量: 数值特征 {len(NUMERIC_FEATURES)} 个, 类别特征 {len(CATEGORICAL_FEATURES)} 个")
    gs.fit(X_train, y_train)
    print(f"最佳参数: {gs.best_params_}")
    best_model = gs.best_estimator_

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n测试集评估报告:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 计算详细指标
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n详细指标:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    # 保存模型
    if save_model:
        joblib.dump(best_model, MODEL_OUT)
        print(f"\n模型已保存为: {MODEL_OUT}")

    # 绘图: 混淆矩阵与 ROC
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-trending', 'Trending'])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix (No Text Features)')
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_no_text.png'), dpi=200, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (No Text Features)')
    plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, 'roc_curve_no_text.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return best_model, (X_test, y_test, y_pred, y_prob, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    })


def extract_feature_importance(model):
    """提取特征重要性（系数）"""
    preprocessor = model.named_steps['preproc']
    clf = model.named_steps['clf']
    
    coefficients = clf.coef_[0]
    
    # 获取特征名称
    feature_names = []
    
    # 数值特征
    for name in NUMERIC_FEATURES:
        feature_names.append(f'num_{name}')
    
    # 类别特征 - 需要从OneHotEncoder获取
    cat_encoder = preprocessor.named_transformers_['cat']
    # 获取类别特征的类别名称
    df = pd.read_csv(DATA_FILE)
    
    # time_period的类别
    time_periods = sorted(df['time_period'].dropna().unique())
    for tp in time_periods:
        feature_names.append(f'cat_time_period_{tp}')
    
    # category的类别
    categories = sorted(df['category'].dropna().unique())
    for cat in categories:
        feature_names.append(f'cat_category_{cat}')
    
    # 创建DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    df_importance['abs_coefficient'] = np.abs(df_importance['coefficient'])
    df_importance = df_importance.sort_values('abs_coefficient', ascending=False)
    
    return df_importance


def predict_unpublished(input_data, model_path=MODEL_OUT):
    """
    对未发布视频做预测。
    input_data: dict 或 pandas.DataFrame，包含需要的特征（不需要title）
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
    X_test, y_test, y_pred, y_prob, metrics = eval_data
    
    print('\n训练与评估完成。')
    print(f"输出文件: {MODEL_OUT}, confusion_matrix_no_text.png, roc_curve_no_text.png")
    
    # 提取特征重要性
    print("\n提取特征重要性...")
    df_importance = extract_feature_importance(model)
    importance_file = os.path.join(SCRIPT_DIR, 'feature_importance_no_text.csv')
    df_importance.to_csv(importance_file, index=False, encoding='utf-8-sig')
    print(f"特征重要性已保存到: {importance_file}")
    
    print("\n最重要的10个特征:")
    for idx, row in df_importance.head(10).iterrows():
        print(f"  {row['feature']:<35} 系数: {row['coefficient']:>10.4f}")
    
    # 保存一个示例预测（从测试集随机取）
    example = X_test.sample(5, random_state=RND)
    example_res = predict_unpublished(example)
    example_file = os.path.join(SCRIPT_DIR, 'example_predictions_no_text.csv')
    example_res.to_csv(example_file, index=False, encoding='utf-8-sig')
    print(f'\n示例预测已保存为: {example_file}')
    
    # 保存评估指标
    metrics_file = os.path.join(SCRIPT_DIR, 'metrics_no_text.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("模型评估指标（不使用文本特征）\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC:   {metrics['auc']:.4f}\n")
    print(f"评估指标已保存到: {metrics_file}")


if __name__ == '__main__':
    main()

