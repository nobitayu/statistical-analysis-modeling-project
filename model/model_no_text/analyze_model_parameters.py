# -*- coding: utf-8 -*-
"""提取并分析不使用文本特征模型的参数及其意义"""
import os
import joblib
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_pipeline_no_text.joblib')
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
    
    # 按特征类型分组
    numeric_features = df_params[df_params['feature'].str.startswith('num_')]
    categorical_features = df_params[df_params['feature'].str.startswith('cat_')]
    
    # 生成详细分析报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("不使用文本特征的模型参数分析")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"模型基本信息:")
    report_lines.append(f"  - 模型类型: 逻辑回归 (Logistic Regression)")
    report_lines.append(f"  - 正则化参数 C: {clf.C}")
    report_lines.append(f"  - 截距 (Intercept): {intercept:.6f}")
    report_lines.append(f"  - 基准概率: {1/(1+np.exp(-intercept)):.4f} (约{1/(1+np.exp(-intercept))*100:.2f}%)")
    report_lines.append(f"  - 总特征数: {len(coefficients)}")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("一、数值特征参数分析")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("数值特征经过标准化处理，系数表示该特征对对数几率的贡献。")
    report_lines.append("正系数表示特征值越大，视频成为热门的概率越高；")
    report_lines.append("负系数表示特征值越大，视频成为热门的概率越低。")
    report_lines.append("")
    report_lines.append("| 特征名称 | 系数 | 绝对值 | 解释 |")
    report_lines.append("|:---|:---:|:---:|:---|")
    
    for idx, row in numeric_features.iterrows():
        feature_name = row['feature'].replace('num_', '')
        coef = row['coefficient']
        abs_coef = row['abs_coefficient']
        
        # 生成解释
        if feature_name == 'tag_count':
            explanation = "标签数量越多，热门概率越低（可能被视为过度营销）"
        elif feature_name == 'tags_in_title':
            explanation = "标题包含标签，热门概率降低"
        elif feature_name == 'hour':
            explanation = "发布时间对热门概率有正向影响"
        elif feature_name == 'question_exclamation_count':
            explanation = "适度的情感表达有助于吸引注意力"
        elif feature_name == 'title_length':
            explanation = "标题过长可能降低吸引力"
        elif feature_name == 'is_weekend':
            explanation = "周末发布的影响很小"
        else:
            explanation = "特征对热门概率的影响"
        
        direction = "增加" if coef > 0 else "降低"
        report_lines.append(f"| {feature_name} | {coef:.4f} | {abs_coef:.4f} | {explanation} |")
    
    report_lines.append("")
    report_lines.append("**数值特征总结:**")
    report_lines.append(f"- 平均系数绝对值: {numeric_features['abs_coefficient'].mean():.4f}")
    report_lines.append(f"- 最大影响: {numeric_features.loc[numeric_features['abs_coefficient'].idxmax(), 'feature'].replace('num_', '')} (系数: {numeric_features['abs_coefficient'].max():.4f})")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("二、类别特征参数分析")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("类别特征通过独热编码处理，系数表示相对于基准类别（第一个类别）的额外贡献。")
    report_lines.append("正系数表示该类别比基准类别更容易成为热门；")
    report_lines.append("负系数表示该类别比基准类别更难成为热门。")
    report_lines.append("")
    
    # 时间段特征
    time_period_features = categorical_features[categorical_features['feature'].str.startswith('cat_time_period_')]
    report_lines.append("### 2.1 时间段特征")
    report_lines.append("")
    report_lines.append("| 时间段 | 系数 | 绝对值 | 解释 |")
    report_lines.append("|:---|:---:|:---:|:---|")
    for idx, row in time_period_features.iterrows():
        time_period = row['feature'].replace('cat_time_period_', '')
        coef = row['coefficient']
        abs_coef = row['abs_coefficient']
        
        if coef > 0:
            explanation = f"在{time_period}发布，热门概率相对较高"
        else:
            explanation = f"在{time_period}发布，热门概率相对较低"
        
        report_lines.append(f"| {time_period} | {coef:.4f} | {abs_coef:.4f} | {explanation} |")
    report_lines.append("")
    
    # 视频类别特征
    category_features = categorical_features[categorical_features['feature'].str.startswith('cat_category_')]
    report_lines.append("### 2.2 视频类别特征")
    report_lines.append("")
    report_lines.append("| 视频类别 | 系数 | 绝对值 | 解释 |")
    report_lines.append("|:---|:---:|:---:|:---|")
    for idx, row in category_features.iterrows():
        category = row['feature'].replace('cat_category_', '')
        coef = row['coefficient']
        abs_coef = row['abs_coefficient']
        
        if coef > 0:
            explanation = f"{category}类别更容易成为热门"
        else:
            explanation = f"{category}类别较难成为热门"
        
        report_lines.append(f"| {category} | {coef:.4f} | {abs_coef:.4f} | {explanation} |")
    report_lines.append("")
    
    report_lines.append("**类别特征总结:**")
    report_lines.append(f"- 平均系数绝对值: {categorical_features['abs_coefficient'].mean():.4f}")
    report_lines.append(f"- 最有利于热门的类别: {category_features.loc[category_features['coefficient'].idxmax(), 'feature'].replace('cat_category_', '')} (系数: {category_features['coefficient'].max():.4f})")
    report_lines.append(f"- 最不利于热门的类别: {category_features.loc[category_features['coefficient'].idxmin(), 'feature'].replace('cat_category_', '')} (系数: {category_features['coefficient'].min():.4f})")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("三、参数解释的实际意义")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 找出最重要的特征
    top_5 = df_params.head(5)
    report_lines.append("### 3.1 最重要的5个特征")
    report_lines.append("")
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        feature_name = row['feature'].replace('num_', '').replace('cat_', '').replace('time_period_', '').replace('category_', '')
        coef = row['coefficient']
        report_lines.append(f"{i}. **{feature_name}** (系数: {coef:.4f})")
        if coef > 0:
            report_lines.append(f"   - 该特征值越大，视频成为热门的概率越高")
        else:
            report_lines.append(f"   - 该特征值越大，视频成为热门的概率越低")
    report_lines.append("")
    
    report_lines.append("### 3.2 业务建议")
    report_lines.append("")
    report_lines.append("基于模型参数分析，可以得出以下业务建议：")
    report_lines.append("")
    
    # 标签相关建议
    tag_coef = numeric_features[numeric_features['feature'] == 'num_tag_count']['coefficient'].values[0]
    if tag_coef < 0:
        report_lines.append(f"1. **标签使用策略**: 标签数量系数为{tag_coef:.4f}，表明过度使用标签会降低视频成为热门的概率。")
        report_lines.append("   建议：适度使用标签，避免标签堆砌，保持内容的自然性。")
        report_lines.append("")
    
    # 类别相关建议
    top_category = category_features.loc[category_features['coefficient'].idxmax()]
    bottom_category = category_features.loc[category_features['coefficient'].idxmin()]
    report_lines.append(f"2. **类别选择**: {top_category['feature'].replace('cat_category_', '')}类别系数最高({top_category['coefficient']:.4f})，")
    report_lines.append(f"   而{bottom_category['feature'].replace('cat_category_', '')}类别系数最低({bottom_category['coefficient']:.4f})。")
    report_lines.append("   建议：根据内容性质选择合适的类别，某些类别天然具有更高的热度潜力。")
    report_lines.append("")
    
    # 发布时间建议
    top_time = time_period_features.loc[time_period_features['coefficient'].idxmax()]
    bottom_time = time_period_features.loc[time_period_features['coefficient'].idxmin()]
    report_lines.append(f"3. **发布时间**: {top_time['feature'].replace('cat_time_period_', '')}时段系数最高({top_time['coefficient']:.4f})，")
    report_lines.append(f"   而{bottom_time['feature'].replace('cat_time_period_', '')}时段系数最低({bottom_time['coefficient']:.4f})。")
    report_lines.append("   建议：选择合适的发布时间可以略微提升视频成为热门的概率。")
    report_lines.append("")
    
    report_lines.append("### 3.3 模型局限性")
    report_lines.append("")
    report_lines.append("由于不使用文本特征，模型无法捕捉标题内容中的语义信息，这是预测视频热度的最重要因素。")
    report_lines.append("因此，本模型更适合作为基线模型或辅助工具，在实际应用中应结合文本特征以获得更好的预测效果。")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("四、完整参数列表")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("| 排名 | 特征名称 | 系数 | 绝对值 | 特征类型 |")
    report_lines.append("|:---:|:---|:---:|:---:|:---|")
    for i, (idx, row) in enumerate(df_params.iterrows(), 1):
        feature_type = "数值" if row['feature'].startswith('num_') else "类别"
        report_lines.append(f"| {i} | {row['feature']} | {row['coefficient']:.4f} | {row['abs_coefficient']:.4f} | {feature_type} |")
    
    # 保存报告
    report_file = os.path.join(SCRIPT_DIR, 'model_parameters_analysis.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("="*80)
    print("模型参数分析完成")
    print("="*80)
    print(f"\n报告已保存到: {report_file}")
    print(f"\n最重要的5个特征:")
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. {row['feature']:<35} 系数: {row['coefficient']:>8.4f}")
    
    return df_params

if __name__ == '__main__':
    df_params = analyze_model_parameters()

