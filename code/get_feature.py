# 脚本主要功能：
# 分类特征编码：类别（category_id）、发布时间（publish_hour）独热编码
# 数值特征衍生：互动率（点赞率、点踩率、评论率）、时间特征（发布星期几）、标题特征（标题长度、大写占比、结尾符号）
import pandas as pd
import numpy as np
import re


df_cleaned = pd.read_csv(r"C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\USvideos_clean.csv")
print(f"修改后清洗数据集形状：{df_cleaned.shape}（行：{df_cleaned.shape[0]}, 列：{df_cleaned.shape[1]}）")

# --------------------------
# 2. 分类特征编码
# --------------------------
print("=== 开始分类特征编码 ===")

# 2.1 视频类别（category_id）：独热编码（One-Hot）
df_onehot = pd.get_dummies(df_cleaned['category_id'], prefix='category', drop_first=False)
df_feat = pd.concat([df_cleaned, df_onehot], axis=1)
df_feat = df_feat.drop(columns=['category_id'])  # 删除原始列
print(f"完成category_id独热编码，新增 {len(df_onehot.columns)} 个类别特征")


# 发布小时→英文时间区间分类
def map_hour_to_english_period(hour):
    """将小时（0-23）映射为英文时间区间"""
    if hour in range(23, 24) or hour in range(0, 6):
        return 'Dawn'       # 凌晨：23-06点
    elif hour in range(6, 11):
        return 'Morning'    # 上午：06-11点
    elif hour in range(11, 14):
        return 'Noon'       # 中午：11-14点
    elif hour in range(14, 18):
        return 'Afternoon'  # 下午：14-18点
    else:  # hour in range(18, 23)
        return 'Evening'    # 晚上：18-23点

# 1. 基于发布小时生成英文时段标签列
df_feat['publish_period_en'] = df_feat['publish_hour'].apply(map_hour_to_english_period)

# 2. 对英文时段标签做独热编码（生成建模用的数值特征）
df_onehot_period_en = pd.get_dummies(
    df_feat['publish_period_en'], 
    prefix='period',  # 特征前缀，最终列名为 period_Dawn、period_Morning 等
    drop_first=False  # 不删除第一列，保留所有时段特征
)

# 3. 将独热编码特征合并到数据集，并删除原始冗余列
df_feat = pd.concat([df_feat, df_onehot_period_en], axis=1)
df_feat = df_feat.drop(columns=['publish_hour', 'publish_period_en'])  # 删除原始小时列和英文标签列
# --------------------------
# 3. 数值特征衍生
# --------------------------
print("=== 开始数值特征衍生 ===")

# 3.1 互动率特征：点赞率、点踩率、评论率（避免views=0导致除零错误）
# （因未做异常值处理，需增加views=0的判断，实际数据中views为热门视频，大概率不为0）
df_feat['like_rate'] = np.where(
    df_feat['views'] > 0,  # 若观看量>0，正常计算
    df_feat['likes'] / df_feat['views'],
    0  # 若观看量=0，点赞率设为0（避免报错）
)
df_feat['dislike_rate'] = np.where(
    df_feat['views'] > 0,
    df_feat['dislikes'] / df_feat['views'],
    0
)
df_feat['comment_rate'] = np.where(
    df_feat['views'] > 0,
    df_feat['comment_count'] / df_feat['views'],
    0
)
print("✅ 新增3个互动率特征：like_rate、dislike_rate、comment_rate")

# 3.2 时间特征：发布星期几（publish_weekday）- 修复datetime格式问题
# 先将字符串格式的publish_date转换为datetime格式
df_feat['publish_date'] = pd.to_datetime(df_feat['publish_date'], errors='coerce')  # errors='coerce'避免异常日期格式报错
# 提取星期几（0=周一~6=周日）
df_feat['publish_weekday'] = df_feat['publish_date'].dt.weekday.fillna(-1).astype(int)
print("✅ 新增1个时间特征：publish_weekday（发布星期几，0=周一~6=周日，-1=日期格式异常）")

# 3.3 标题长度特征：视频标题长度（title_length）
df_feat['title_length'] = df_feat['title'].str.len()
print("✅ 新增1个标题特征：title_length（标题长度）")

# 3.4 标题大写占比：大写字符数/标题总长度（title_upper_ratio）
def count_upper_chars(text):
    if pd.isna(text):  # 应对可能的title缺失
        return 0
    return sum(1 for char in text if char.isupper())

df_feat['title_upper_count'] = df_feat['title'].apply(count_upper_chars)
# 增加title_length=0的保护
df_feat['title_upper_ratio'] = np.where(
    df_feat['title_length'] > 0,
    df_feat['title_upper_count'] / df_feat['title_length'],
    0
)
print("✅ 新增1个标题特征：title_upper_ratio（标题大写占比）")

# 3.5 标题结尾符号：是否以？或！结尾（title_ends_with_punct）
def check_ending_punct(text):
    if pd.isna(text):
        return 0
    return 1 if text.strip().endswith(('?', '!')) else 0

df_feat['title_ends_with_punct'] = df_feat['title'].apply(check_ending_punct)
print("✅ 新增1个标题特征：title_ends_with_punct（标题是否以？/！结尾）")

# --------------------------
# 5. 验证与保存
# --------------------------
print(f"\n=== 特征工程完成总结 ===")
print(f"最终特征数据集形状：{df_feat.shape}（行：{df_feat.shape[0]}, 列：{df_feat.shape[1]}）")
print(f"新增特征总数：{df_feat.shape[1] - df_cleaned.shape[1]}")

# 保存到本地
save_path = r"C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\USvideos_featured.csv"
df_feat.to_csv(save_path, index=False)
print(f"\n特征工程后的数据已保存至：{save_path}")