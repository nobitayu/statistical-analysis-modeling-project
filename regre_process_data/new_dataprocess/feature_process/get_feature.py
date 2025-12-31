# 脚本主要功能：
# 分类特征编码：类别（categoryId）、发布时间（publishedAt）独热编码
# 数值特征衍生：互动率（点赞率、评论率）、时间特征（是否发布在周末，bool值）、标题特征（标题长度、大写占比、是否包含问号或感叹号）

import pandas as pd
import numpy as np
import re
import os

# 定义文件路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_Sampled.csv"
output_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_Featured.csv"

def main():
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    # 读取数据，尝试常见编码
    try:
        df_cleaned = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_cleaned = pd.read_csv(input_file_path, encoding='latin1')
        
    print(f"原始数据集形状：{df_cleaned.shape}（行：{df_cleaned.shape[0]}, 列：{df_cleaned.shape[1]}）")

    # 预处理：转换时间格式
    # publishedAt 格式类似 2022/4/17 17:08:05 或 2022/4/17 17:08
    df_cleaned['publishedAt'] = pd.to_datetime(df_cleaned['publishedAt'], errors='coerce')
    
    # --------------------------
    # 2. 分类特征编码
    # --------------------------
    print("=== 开始分类特征编码 ===")
    
    # 初始化特征数据集，保留原始数据
    df_feat = df_cleaned.copy()

    # 2.1 视频类别（categoryId）：独热编码（One-Hot）
    if 'categoryId' in df_feat.columns:
        df_onehot = pd.get_dummies(df_feat['categoryId'], prefix='category', drop_first=False)
        df_feat = pd.concat([df_feat, df_onehot], axis=1)
        # df_feat = df_feat.drop(columns=['categoryId'])  # 用户要求不要删掉原本文件的列
        print(f"完成 categoryId 独热编码，新增 {len(df_onehot.columns)} 个类别特征")
    else:
        print("Warning: 'categoryId' column not found.")

    # 2.2 发布时间（publishedAt）：提取小时 -> 时段 -> 独热编码
    # 发布小时→英文时间区间分类
    def map_hour_to_english_period(hour):
        """将小时（0-23）映射为英文时间区间"""
        if pd.isna(hour):
            return 'Unknown'
        hour = int(hour)
        if hour in range(0, 6):
            return 'Dawn'       # 凌晨：0-6点
        elif hour in range(6,12):
            return 'Morning'    # 上午：6-12点
        elif hour in range(12,18):
            return 'Afternoon'  # 下午：12-18点
        else:  # hour in range(18, 24)
            return 'Evening'    # 晚上：18-0点

    # 提取小时
    df_feat['publish_hour'] = df_feat['publishedAt'].dt.hour
    
    # 基于发布小时生成英文时段标签列
    df_feat['publish_period_en'] = df_feat['publish_hour'].apply(map_hour_to_english_period)
    
    # 对英文时段标签做独热编码
    df_onehot_period_en = pd.get_dummies(
        df_feat['publish_period_en'], 
        prefix='period',  # 特征前缀
        drop_first=False
    )
    
    # 将独热编码特征合并到数据集
    df_feat = pd.concat([df_feat, df_onehot_period_en], axis=1)
    # df_feat = df_feat.drop(columns=['publish_hour', 'publish_period_en']) # 用户要求不要删掉原本文件的列
    print(f"完成发布时间独热编码，新增 {len(df_onehot_period_en.columns)} 个时间段特征")

    # --------------------------
    # 3. 数值特征衍生
    # --------------------------
    print("=== 开始数值特征衍生 ===")

    # 3.1 互动率特征：点赞率、评论率
    # view_count 为分母
    # 确保 view_count 为数值
    df_feat['view_count'] = pd.to_numeric(df_feat['view_count'], errors='coerce').fillna(0)
    df_feat['likes'] = pd.to_numeric(df_feat['likes'], errors='coerce').fillna(0)
    df_feat['comment_count'] = pd.to_numeric(df_feat['comment_count'], errors='coerce').fillna(0)

    df_feat['like_rate'] = np.where(
        df_feat['view_count'] > 0,
        df_feat['likes'] / df_feat['view_count'],
        0
    )
    
    df_feat['comment_rate'] = np.where(
        df_feat['view_count'] > 0,
        df_feat['comment_count'] / df_feat['view_count'],
        0
    )
    print("✅ 新增2个互动率特征：like_rate、comment_rate")

    # 3.2 时间特征：是否发布在周末（bool值）
    # weekday: 0=Monday, 6=Sunday
    # 周末: 5 (Saturday), 6 (Sunday)
    df_feat['is_weekend'] = df_feat['publishedAt'].dt.weekday.isin([5, 6])
    print("✅ 新增1个时间特征：is_weekend（是否发布在周末）")

    # 3.3 标题特征
    # 确保 title 为字符串
    df_feat['title'] = df_feat['title'].astype(str)

    # 标题长度
    df_feat['title_length'] = df_feat['title'].str.len()
    
    # 标题大写占比
    def count_upper_chars(text):
        if pd.isna(text):
            return 0
        return sum(1 for char in text if char.isupper())

    df_feat['title_upper_count'] = df_feat['title'].apply(count_upper_chars)
    
    df_feat['title_upper_ratio'] = np.where(
        df_feat['title_length'] > 0,
        df_feat['title_upper_count'] / df_feat['title_length'],
        0
    )

    # 标题是否包含问号或感叹号
    def check_has_punct(text):
        if pd.isna(text):
            return False
        # 只要包含 ? 或 ! 即为 True，不一定要在结尾
        return any(char in ['?', '!', '？', '！'] for char in text)
        # 如果需求严格是"结尾符号"，则用 text.strip().endswith(('?', '!'))
        # 用户需求文字："是否包含问号或感叹号"，倾向于包含即可。
        # 但看原代码是 title_ends_with_punct，为了更符合通常的“标题党”检测，包含更有意义。
        # 让我们按照字面意思“包含”来做。

    df_feat['title_has_punct'] = df_feat['title'].apply(check_has_punct)
    
    print("✅ 新增3个标题特征：title_length, title_upper_ratio, title_has_punct")

    # --------------------------
    # 5. 验证与保存
    # --------------------------
    print(f"\n=== 特征工程完成总结 ===")
    print(f"最终特征数据集形状：{df_feat.shape}（行：{df_feat.shape[0]}, 列：{df_feat.shape[1]}）")
    print(f"新增特征总数：{df_feat.shape[1] - df_cleaned.shape[1]}")

    # 保存到本地
    df_feat.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"\n特征工程后的数据已保存至：{output_file_path}")

if __name__ == "__main__":
    main()