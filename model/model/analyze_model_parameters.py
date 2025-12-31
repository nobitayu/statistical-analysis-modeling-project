# -*- coding: utf-8 -*-
"""提取并分析模型的参数及其意义"""
import os
import joblib
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_pipeline.joblib')
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'process', 'youtube_data_balanced_5000.csv')

NUMERIC_FEATURES = ['title_length', 'question_exclamation_count', 'tag_count', 'hour', 'tags_in_title', 'is_weekend']
CATEGORICAL_FEATURES = ['time_period', 'category']

def analyze_model_parameters():
    """分析模型参数并生成详细说明"""
    print("加载模型...")
    model = joblib.load(MODEL_PATH)
    
    # 获取逻辑回归分类器
    clf = model.named_steps['clf']
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    print(f"模型类型: {type(clf).__name__}")
    print(f"正则化参数 C: {clf.C}")
    print(f"系数数量: {len(coefficients)}")
    print(f"截距: {intercept:.6f}\n")
    
    # 获取特征名称
    feature_names = []
    
    # 数值特征
    for name in NUMERIC_FEATURES:
        feature_names.append(f'num_{name}')
    
    # 类别特征
    df = pd.read_csv(DATA_FILE)
    time_periods = sorted(df['time_period'].dropna().unique())
    for tp in time_periods:
        feature_names.append(f'cat_time_period_{tp}')
    
    categories = sorted(df['category'].dropna().unique())
    for cat in categories:
        feature_names.append(f'cat_category_{cat}')
    
    # 创建DataFrame
    df_params = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    df_params['abs_coefficient'] = np.abs(df_params['coefficient'])
    df_params = df_params.sort_values('abs_coefficient', ascending=False)
    
    # 保存参数到CSV文件
    output_file = os.path.join(SCRIPT_DIR, 'model_parameters.csv')
    df_params.to_csv(output_file, index=False, encoding='utf-8')
    
    print("="*80)
    print("模型参数提取完成")
    print("="*80)
    print(f"\n模型基本信息:")
    print(f"  - 模型类型: {type(clf).__name__}")
    print(f"  - 正则化参数 C: {clf.C}")
    print(f"  - 截距 (Intercept): {intercept:.6f}")
    print(f"  - 总特征数: {len(coefficients)}")
    print(f"\n参数已保存到: {output_file}")

    
    return df_params

if __name__ == '__main__':
    df_params = analyze_model_parameters()

