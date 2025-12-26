#脚本主要功能：
#缺失值处理：1. 删除订阅数缺失行；2. 文本特征用空字符串填充
#数据格式统一：时间特征转换为datetime格式，布尔特征转换为0/1数值
#删除唯一标识
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\USvideos_modified.csv")
print(f"原始数据集形状：{df.shape}（行：{df.shape[0]}, 列：{df.shape[1]}）")

# --------------------------
# 2. 缺失值处理（核心步骤）
# --------------------------
print("\n=== 缺失值统计（处理前）===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失值数量': missing_values,
    '缺失比例(%)': missing_percentage.round(2)
})
print(missing_df[missing_df['缺失值数量'] > 0])  # 只显示有缺失的列

# 2.1 关键数值特征：删除订阅数（subscriber）缺失的行（占比0.48%，影响小）
df = df.dropna(subset=['subscriber'])
print(f"\n删除subscriber缺失行后，数据集形状：{df.shape}")

# 2.2 文本特征：填充标签（tags）和描述（description）缺失值（用空字符串，不影响数值建模）
df['tags'] = df['tags'].fillna('')
df['description'] = df['description'].fillna('')

print("\n=== 缺失值统计（处理后）===")
print(df.isnull().sum()[df.isnull().sum() > 0])  # 验证无缺失值残留

# # --------------------------
# # 3. 异常值处理（基于3σ准则的缩尾处理）
# # --------------------------
# # 定义缩尾处理函数：将超过3个标准差的异常值替换为3σ边界值
# def winsorize_by_3sigma(data, column):
#     mean_val = data[column].mean()
#     std_val = data[column].std()
#     # 计算上下边界（3σ准则）
#     upper_bound = mean_val + 3 * std_val
#     lower_bound = mean_val - 3 * std_val
#     # 缩尾处理（避免删除样本，保留爆款视频信息）
#     data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
#     return data

# # 对核心数值特征（观看量、互动数据、订阅数）进行异常值处理
# numeric_cols = ['views', 'likes', 'dislikes', 'comment_count', 'subscriber']
# print(f"\n=== 异常值处理（对 {numeric_cols} 进行3σ缩尾）===")
# for col in numeric_cols:
#     before_outliers = ((df[col] > df[col].mean() + 3 * df[col].std()) | 
#                       (df[col] < df[col].mean() - 3 * df[col].std())).sum()
#     df = winsorize_by_3sigma(df, col)
#     after_outliers = ((df[col] > df[col].mean() + 3 * df[col].std()) | 
#                      (df[col] < df[col].mean() - 3 * df[col].std())).sum()
#     print(f"{col}：处理前异常值数量 {before_outliers} → 处理后 {after_outliers}")

# --------------------------
# 4. 数据格式统一（时间特征+布尔特征）
# --------------------------
# 4.1 时间特征：转换为datetime格式,便于后续提取星期、月份等特征
df['publish_date'] = pd.to_datetime(df['publish_date'])
df['last_trending_date'] = pd.to_datetime(df['last_trending_date'])
print(f"\n=== 时间格式转换 ===")
print(f"publish_date 数据类型：{df['publish_date'].dtype}")
print(f"last_trending_date 数据类型：{df['last_trending_date'].dtype}")

# 4.2 布尔特征：转换为0/1数值
bool_cols = ['comments_disabled', 'ratings_disabled','tag_appeared_in_title']
df[bool_cols] = df[bool_cols].astype(int)  # True→1，False→0
print(f"\n=== 布尔特征转换 ===")
print(f"布尔特征 {bool_cols} 已转换为 0/1 数值型")

# 删除的列：唯一标识（video_id）
drop_cols = ['video_id']
df_cleaned = df.drop(columns=drop_cols)

# --------------------------
# 6. 保存清洗后的数据（供后续特征工程和建模使用）
# --------------------------
df_cleaned.to_csv('USvideos_clean.csv', index=False)
print(f"\n清洗后的数据已保存")