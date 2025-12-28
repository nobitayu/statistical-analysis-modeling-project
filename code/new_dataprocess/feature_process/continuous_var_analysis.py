# 脚本主要功能：
# 1. 连续变量分析：读取特征工程后的数据集，对关键连续变量进行统计分析
# 2. 数据可视化：绘制变量的直方图、Q-Q图等，展示数据分布情况
# 3. 异常值检测：识别并分析数据中的离群点和极端值

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# 配置中文字体，避免绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 增加：调整全局字体大小 ===
plt.rcParams['font.size'] = 14          # 全局默认字体大小
plt.rcParams['axes.titlesize'] = 16     # 标题字体大小
plt.rcParams['axes.labelsize'] = 14     # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12    # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12    # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12    # 图例字体大小
# ==============================

# 文件路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_All_Features.csv"
output_dir = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\analysis_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    if not os.path.exists(input_file_path):
        print(f"Error: File not found at {input_file_path}")
        return

    # 读取数据
    try:
        # 尝试常见编码
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 读取失败，尝试 Latin1...")
            df = pd.read_csv(input_file_path, encoding='latin1')
            
        print(f"成功读取数据，形状：{df.shape}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    # 待处理变量列表
    # 注意：tag_count 在前一步生成的可能是 tags_count，需确认
    # 检查实际列名
    target_cols = [
        'view_count', 'likes', 'comment_count', 'title_length', 
        'tags_count', # 假设前一步生成的是 tags_count
        'channel_activity', 'channel_avg_views', 'channel_avg_like_rate', 
        'channel_avg_comment_count', 'channel_name_len', 'desc_length'
    ]
    
    # 检查列是否存在，如果 tags_count 不存在尝试 tag_count
    if 'tags_count' not in df.columns and 'tag_count' in df.columns:
        target_cols = [c if c != 'tags_count' else 'tag_count' for c in target_cols]
    
    existing_cols = [col for col in target_cols if col in df.columns]
    missing_cols = set(target_cols) - set(existing_cols)
    if missing_cols:
        print(f"警告：以下列在数据集中未找到，将被跳过：{missing_cols}")
    
    # 确保均为数值类型
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. 描述性统计
    print("\n=== 1. 描述性统计 ===")
    stats_df = pd.DataFrame(index=existing_cols, columns=['Mean', 'Std', 'Median', 'Skewness', 'Kurtosis', 'Remark'])
    
    for col in existing_cols:
        data = df[col]
        stats_df.loc[col, 'Mean'] = data.mean()
        stats_df.loc[col, 'Std'] = data.std()
        stats_df.loc[col, 'Median'] = data.median()
        stats_df.loc[col, 'Skewness'] = data.skew()
        stats_df.loc[col, 'Kurtosis'] = data.kurt()
        
        remark = []
        if abs(data.skew()) > 1:
            remark.append("显著偏态")
        if abs(data.kurt()) > 3:
            remark.append("尖峰/平峰")
        stats_df.loc[col, 'Remark'] = ", ".join(remark) if remark else "正常"

    print(stats_df)
    stats_df.to_csv(os.path.join(output_dir, "descriptive_stats.csv"))
    print(f"描述性统计结果已保存至 {os.path.join(output_dir, 'descriptive_stats.csv')}")

    # 2. 可视化诊断（原始分布）
    print("\n=== 2. 可视化诊断（原始分布） ===")
    plot_distribution(df, existing_cols, suffix="_original")

    # 3. 分布矫正（对数变换）
    print("\n=== 3. 分布矫正（对数变换） ===")
    df_transformed = df.copy()
    transformed_cols = []
    
    for col in existing_cols:
        new_col_name = f"log_{col}"
        # ln(x + 1) 变换
        # 对于可能有负数或0的情况，加1通常是安全的处理非负数的方式
        # 如果有负数，可能需要先平移
        if df[col].min() < 0:
            print(f"警告：列 {col} 存在负值 (min={df[col].min()})，将平移后再取对数")
            shift = abs(df[col].min()) + 1
            df_transformed[new_col_name] = np.log(df[col] + shift)
        else:
            df_transformed[new_col_name] = np.log(df[col] + 1)
        
        transformed_cols.append(new_col_name)

    # 4. 变换后验证
    print("\n=== 4. 变换后验证 ===")
    # 重新计算描述性统计
    stats_trans_df = pd.DataFrame(index=transformed_cols, columns=['Mean', 'Std', 'Median', 'Skewness', 'Kurtosis', 'Remark'])
    for col in transformed_cols:
        data = df_transformed[col]
        stats_trans_df.loc[col, 'Mean'] = data.mean()
        stats_trans_df.loc[col, 'Std'] = data.std()
        stats_trans_df.loc[col, 'Median'] = data.median()
        stats_trans_df.loc[col, 'Skewness'] = data.skew()
        stats_trans_df.loc[col, 'Kurtosis'] = data.kurt()
        
        remark = []
        if abs(data.skew()) > 1:
            remark.append("显著偏态")
        stats_trans_df.loc[col, 'Remark'] = ", ".join(remark) if remark else "改善"
    
    print("变换后统计量：")
    print(stats_trans_df)
    stats_trans_df.to_csv(os.path.join(output_dir, "transformed_stats.csv"))

    # 绘制变换后的分布图
    plot_distribution(df_transformed, transformed_cols, suffix="_transformed")

    # Shapiro-Wilk 检验 (对 view_count 对应的 log_view_count)
    if 'log_view_count' in df_transformed.columns:
        print("\n=== Shapiro-Wilk 正态性检验 (因变量 log_view_count) ===")
        # Shapiro-Wilk 对样本量有限制（通常 N < 5000），若样本量过大，p值可能过于敏感
        # 我们进行随机抽样 5000 个进行检验（如果数据量 > 5000）
        data_for_test = df_transformed['log_view_count']
        if len(data_for_test) > 5000:
            data_for_test = data_for_test.sample(5000, random_state=42)
            print("注：样本量 > 5000，随机抽取 5000 个样本进行检验")
        
        stat, p_value = stats.shapiro(data_for_test)
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4g}")
        if p_value > 0.05:
            print("结论：数据服从正态分布 (p > 0.05)")
        else:
            print("结论：数据显著偏离正态分布 (p <= 0.05)")
            print("提示：在大样本下，Shapiro-Wilk 检验非常敏感，即使轻微偏离也会拒绝原假设。建议结合 Q-Q 图和直方图综合判断。")

    # 保存变换后的数据
    output_csv_path = os.path.join(output_dir, "New_Youtube_Videos_2022_Transformed.csv")
    df_transformed.to_csv(output_csv_path, index=False)
    print(f"\n变换后的数据已保存至：{output_csv_path}")

def plot_distribution(df, cols, suffix=""):
    """绘制直方图和 Q-Q 图"""
    n_cols = 2
    n_rows = len(cols)
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cols):
        # 直方图
        ax1 = fig.add_subplot(n_rows, n_cols, 2 * i + 1)
        sns.histplot(df[col], kde=True, ax=ax1)
        ax1.set_title(f'Histogram of {col} {suffix}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frequency')
        
        # Q-Q 图
        ax2 = fig.add_subplot(n_rows, n_cols, 2 * i + 2)
        stats.probplot(df[col], dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot of {col} {suffix}')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"distribution_plots{suffix}.png")
    plt.savefig(save_path)
    print(f"分布图已保存至 {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
