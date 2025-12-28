# 脚本主要功能：
# 1. 文本特征提取：基于channelTitle提取频道活跃度、频道平均播放量/点赞率/评论数
# 2. 标签特征提取：计算视频标签数量（tags_count）
# 3. 描述特征提取：计算视频描述长度（desc_length）及关键词出现频率

import pandas as pd
import numpy as np
from io import StringIO
import os

# 定义输入和输出路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_Featured.csv"
output_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_All_Features.csv"

def main():
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    try:
        # 读取数据，尝试常见编码
        try:
            df_final = pd.read_csv(input_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_final = pd.read_csv(input_file_path, encoding='latin1')
            
        print(f"✅ 文件读取完成！数据集形状：{df_final.shape}（行：{df_final.shape[0]}, 列：{df_final.shape[1]}）")
        
        # 验证必要列是否存在
        required_cols = ['channelTitle', 'tags', 'description']
        missing_cols = [col for col in required_cols if col not in df_final.columns]
        if missing_cols:
            raise ValueError(f"数据集缺少必要列：{missing_cols}，请重新生成包含这些列的 CSV 文件")

        # --------------------------
        # 2. channelTitle特征提取
        # --------------------------
        print("=== 1. 提取 channelTitle 特征 ===")

        # 2.1 频道活跃度：该频道在数据集中的视频数量
        channel_activity = df_final['channelTitle'].value_counts().to_dict()
        df_final['channel_activity'] = df_final['channelTitle'].map(channel_activity)

        # 2.2 频道平均表现：该频道所有视频的平均观看量、平均点赞率
        # 需要确保 views, like_rate, comment_rate 存在
        # 根据上一步特征工程，view_count, like_rate, comment_rate 应该存在
        # 注意：原代码用的是 'views'，新数据集是 'view_count'
        
        # 确保数值列为数值类型
        cols_to_numeric = ['view_count', 'like_rate', 'comment_rate']
        for col in cols_to_numeric:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
            else:
                print(f"Warning: Column {col} not found for aggregation.")

        channel_avg = df_final.groupby('channelTitle').agg({
            'view_count': 'mean',      # 频道平均观看量
            'like_rate': 'mean',       # 频道平均点赞率
            'comment_rate': 'mean'     # 频道平均评论率
        }).reset_index()
        channel_avg.columns = ['channelTitle', 'channel_avg_views', 'channel_avg_like_rate', 'channel_avg_comment_count']
        df_final = pd.merge(df_final, channel_avg, on='channelTitle', how='left')

        # 2.3 频道名属性：长度、是否含数字/特殊符号
        df_final['channel_name_len'] = df_final['channelTitle'].astype(str).str.len()  # 长度
        df_final['channel_has_digit'] = df_final['channelTitle'].astype(str).str.contains(r'\d', regex=True).fillna(0).astype(int)  # 含数字
        df_final['channel_has_special'] = df_final['channelTitle'].astype(str).str.contains(r'[&@#$%\-_]', regex=True).fillna(0).astype(int)  # 含特殊符号

        # 删除原始频道名列
        df_final = df_final.drop(columns=['channelTitle'])
        print(f"✅ 新增 6 个 channelTitle 特征：channel_activity、channel_avg_views、channel_avg_like_rate、channel_name_len、channel_has_digit、channel_has_special\n")

        # --------------------------
        # 3. tags特征提取
        # --------------------------
        print("=== 2. 提取 tags 特征 ===")

        # 确保 tags 列为字符串
        df_final['tags'] = df_final['tags'].astype(str)
        
        # 计算 tags_count (原脚本似乎假设有这个列，或者需要新生成)
        # 这里我们生成 tags_count
        df_final['tags_count'] = df_final['tags'].apply(lambda x: len(x.split('|')) if x and x.lower() != 'nan' and x.strip() != '' else 0)

        # 3.1 标签密度（标签数/标题长度）
        # 确保 title_length 存在
        if 'title_length' not in df_final.columns:
             df_final['title_length'] = df_final['title'].astype(str).str.len()

        df_final['tag_density'] = np.where(
            df_final['title_length'] > 0,
            df_final['tags_count'] / df_final['title_length'],
            0
        )  # 标签密度

        # 3.2 热门标签占比：统计前10%热门标签，计算每个视频的热门标签比例
        all_tags = []
        for tag_str in df_final['tags']:
            if str(tag_str).strip() != '' and str(tag_str).lower() != 'nan':
                all_tags.extend(str(tag_str).split('|'))
        
        # 计算热门标签（出现频率前0.5%）
        if len(all_tags) > 0:  # 避免无标签数据报错
            tag_freq = pd.Series(all_tags).value_counts()
            popular_threshold = int(len(tag_freq) * 0.001)  
            # 确保阈值至少为1（避免标签总数过少时threshold=0）
            popular_threshold = max(popular_threshold, 1)
            # 提取前10%热门标签（含标签名和出现次数）
            popular_tags_df = tag_freq.head(popular_threshold).reset_index()
            # pandas版本差异，reset_index后列名可能不同，统一下
            popular_tags_df.columns = ['热门标签名', '出现次数']
            popular_tags = set(popular_tags_df['热门标签名'])  # 用于后续计算占比
            
            # 打印前10%热门标签的名称和出现次数
            print(f"✅ 数据集中共有 {len(tag_freq)} 个标签，前0.1%热门标签（共 {popular_threshold} 个）如下：")
            # print("-" * 50)
            # print(popular_tags_df.to_string(index=False)) 
            # print("-" * 50)
            
            # 计算热门标签占比
            def calc_popular_ratio(tag_str):
                if str(tag_str).strip() == '' or str(tag_str).lower() == 'nan':
                    return 0
                tag_list = str(tag_str).split('|')
                popular_num = sum(1 for tag in tag_list if tag in popular_tags)
                return popular_num / len(tag_list)
            df_final['popular_tag_ratio'] = df_final['tags'].apply(calc_popular_ratio)
        else:
            df_final['popular_tag_ratio'] = 0  # 无标签时设为0
            print("⚠️  数据集中无有效标签，热门标签占比设为0")
            
        # --------------------------
        # 4. description特征提取
        # --------------------------
        print("=== 3. 提取 description 特征 ===")

        # 确保 description 为字符串
        df_final['description'] = df_final['description'].astype(str)

        # 4.1 描述基础统计：描述长度、是否含实用信息（链接、时间戳）
        df_final['desc_length'] = df_final['description'].apply(lambda x: len(str(x)) if pd.notna(x) and str(x).lower() != 'nan' else 0)  # 描述长度
        
        # 描述是否含 YouTube 链接
        df_final['desc_has_youtube_link'] = df_final['description'].str.contains(r'youtube\.com|youtu\.be', regex=True, na=False).astype(int)
        
        # 描述是否含时间戳
        df_final['desc_has_timestamp'] = df_final['description'].str.contains(r'\d{1,2}:\d{2}', regex=True, na=False).astype(int)

        # 4.2 描述关键词数量：统计视频相关关键词出现次数
        key_words = ['tutorial', 'review', 'vlog', 'recipe', 'guide', 'tips', 'how to', 'best', 'top', 'learn', 'make', 'easy']
        def count_keywords(desc):
            if pd.isna(desc) or str(desc).strip() == '' or str(desc).lower() == 'nan':
                return 0
            desc_lower = str(desc).lower()
            return sum(1 for word in key_words if word in desc_lower)
        
        df_final['desc_keyword_count'] = df_final['description'].apply(count_keywords)

        # 删除原始描述列
        df_final = df_final.drop(columns=['description'])
        print(f"✅ 新增 4 个 description 特征：desc_length、desc_has_youtube_link、desc_has_timestamp、desc_keyword_count\n")

        print("=== 文本特征提取完成总结 ===")
        df_final.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 最终数据集已保存至：{output_file_path}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
