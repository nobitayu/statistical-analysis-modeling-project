# 脚本主要功能：
# 1. 分布改善验证：对比原始数据与变换（如Log变换）后的数据分布差异
# 2. 量化指标计算：计算偏度（Skewness）和Shapiro-Wilk W统计量
# 3. 效果评估：通过指标变化证明特征变换对数据正态性的改善效果

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_All_Features.csv"
output_dir = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\analysis_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    # 1. 读取数据
    try:
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file_path, encoding='latin1')
    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    # 待分析变量
    target_cols = [
        'view_count', 'likes', 'comment_count', 'title_length', 
        'tags_count', 'channel_activity', 'channel_avg_views', 
        'channel_avg_like_rate', 'channel_avg_comment_count', 
        'channel_name_len', 'desc_length'
    ]
    
    # 检查列是否存在（兼容 tags_count/tag_count）
    if 'tags_count' not in df.columns and 'tag_count' in df.columns:
        target_cols = [c if c != 'tags_count' else 'tag_count' for c in target_cols]
    
    existing_cols = [col for col in target_cols if col in df.columns]
    
    # 确保数值类型
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. 计算对比指标
    results = []
    
    print(f"{'变量名':<25} | {'原始偏度':<10} | {'变换后偏度':<10} | {'偏度改善率':<10} | {'原始W值':<10} | {'变换后W值':<10}")
    print("-" * 100)

    for col in existing_cols:
        # 原始数据
        original_data = df[col]
        skew_org = original_data.skew()
        kurt_org = original_data.kurt()
        
        # Shapiro-Wilk (采样 5000)
        sample_org = original_data.sample(min(5000, len(original_data)), random_state=42)
        w_org, p_org = stats.shapiro(sample_org)
        
        # 对数变换
        if original_data.min() < 0:
            shift = abs(original_data.min()) + 1
            trans_data = np.log(original_data + shift)
        else:
            trans_data = np.log(original_data + 1)
            
        skew_trans = trans_data.skew()
        kurt_trans = trans_data.kurt()
        
        sample_trans = trans_data.sample(min(5000, len(trans_data)), random_state=42)
        w_trans, p_trans = stats.shapiro(sample_trans)
        
        # 计算改善程度
        # 偏度改善：看绝对值是否变小
        skew_imp = (abs(skew_org) - abs(skew_trans)) / abs(skew_org) * 100
        # W值改善：看数值是否增加（越接近1越好）
        w_imp = (w_trans - w_org) / w_org * 100
        
        results.append({
            'Variable': col,
            'Skew_Org': skew_org,
            'Skew_Trans': skew_trans,
            'Skew_Imp_Pct': skew_imp,
            'Kurt_Org': kurt_org,
            'Kurt_Trans': kurt_trans,
            'W_Org': w_org,
            'W_Trans': w_trans,
            'W_Imp_Pct': w_imp
        })
        
        print(f"{col:<25} | {skew_org:<10.2f} | {skew_trans:<10.2f} | {skew_imp:<10.1f}% | {w_org:<10.4f} | {w_trans:<10.4f}")

    # 保存对比表
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, "distribution_improvement_metrics.csv"), index=False)
    
    # 3. 绘制核心对比图 (精选几个关键变量)
    # 选择 Skewness 改善最显著的 Top 4 变量进行绘图
    top_improved = df_res.sort_values('Skew_Imp_Pct', ascending=False).head(4)['Variable'].tolist()
    # 确保 view_count 在里面，如果不在则强制加入（因为它是最重要的因变量）
    if 'view_count' in existing_cols and 'view_count' not in top_improved:
        top_improved[0] = 'view_count'
        
    plot_comparison(df, top_improved, output_dir)
    
    print("\n" + "="*50)
    print("证明改善的依据：")
    print("1. 偏度（Skewness）：变换后的偏度绝对值显著降低，更接近 0（正态分布偏度为0）。")
    print("2. 峰度（Kurtosis）：变换后的峰度通常会降低，缓和尖峰或厚尾现象。")
    print("3. Shapiro-Wilk W统计量：变换后的 W 值显著增加并接近 1（W=1 表示完全正态）。")
    print("4. 可视化直方图：变换后的分布曲线更接近对称的钟形曲线。")
    print("="*50)

def plot_comparison(df, cols, output_dir):
    fig, axes = plt.subplots(len(cols), 2, figsize=(14, 4 * len(cols)))
    
    for i, col in enumerate(cols):
        # 原始数据
        original = df[col]
        # 变换数据
        if original.min() < 0:
            shift = abs(original.min()) + 1
            transformed = np.log(original + shift)
        else:
            transformed = np.log(original + 1)
            
        # 左图：原始
        sns.histplot(original, kde=True, ax=axes[i, 0], color='skyblue')
        axes[i, 0].set_title(f'{col} (Original)\nSkew: {original.skew():.2f}')
        
        # 右图：变换后
        sns.histplot(transformed, kde=True, ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'Log({col}) (Transformed)\nSkew: {transformed.skew():.2f}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "key_variables_comparison.png"))
    print(f"\n关键变量对比图已保存至: {os.path.join(output_dir, 'key_variables_comparison.png')}")

if __name__ == "__main__":
    main()
