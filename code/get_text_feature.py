import pandas as pd
import numpy as np
from io import StringIO

feat_path = r"C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\USvideos_featured.csv"

try:
    # 1. 二进制模式读取（不涉及编码，100%获取文件内容）
    with open(feat_path, 'rb') as f:
        file_bytes = f.read()
    
    # 2. GBK解码（专门处理0xa1这类GBK特殊字符），特殊字符自动替换
    file_content = file_bytes.decode('gbk', errors='replace')
    print(f"✅ GBK编码解码成功（已处理特殊字符）")
    
    # 3. 转为UTF-8格式的内存文件流，pandas直接读取
    utf8_stream = StringIO(file_content.encode('utf-8').decode('utf-8'))
    df_final = pd.read_csv(utf8_stream)  # 读取内存流，而非原始文件
    
    print(f"✅ 文件读取完成！数据集形状：{df_final.shape}（行：{df_final.shape[0]}, 列：{df_final.shape[1]}）")
except Exception as e:
    raise ValueError(f"读取失败！错误详情：{str(e)}\n请检查：1.文件路径是否正确 2.文件是否损坏")

# 验证必要列是否存在
required_cols = ['channel_title', 'tags', 'description']
missing_cols = [col for col in required_cols if col not in df_final.columns]
if missing_cols:
    raise ValueError(f"数据集缺少必要列：{missing_cols}，请重新生成包含这些列的 USvideos_featured.csv")

# --------------------------
# 2. channel_title特征提取
# --------------------------
print("=== 1. 提取 channel_title 特征 ===")

# 2.1 频道活跃度：该频道在数据集中的视频数量
channel_activity = df_final['channel_title'].value_counts().to_dict()
df_final['channel_activity'] = df_final['channel_title'].map(channel_activity)

# 2.2 频道平均表现：该频道所有视频的平均观看量、平均点赞率
channel_avg = df_final.groupby('channel_title').agg({
    'views': 'mean',          # 频道平均观看量
    'like_rate': 'mean',       # 频道平均点赞率
    'comment_rate': 'mean'   # 频道平均评论率
}).reset_index()
channel_avg.columns = ['channel_title', 'channel_avg_views', 'channel_avg_like_rate', 'channel_avg_comment_count']
df_final = pd.merge(df_final, channel_avg, on='channel_title', how='left')

# 2.3 频道名属性：长度、是否含数字/特殊符号
df_final['channel_name_len'] = df_final['channel_title'].str.len()  # 长度
df_final['channel_has_digit'] = df_final['channel_title'].str.contains(r'\d', regex=True).fillna(0).astype(int)  # 含数字
df_final['channel_has_special'] = df_final['channel_title'].str.contains(r'[&@#$%\-_]', regex=True).fillna(0).astype(int)  # 含特殊符号

# 删除原始频道名列
df_final = df_final.drop(columns=['channel_title'])
print(f"✅ 新增 6 个 channel_title 特征：channel_activity、channel_avg_views、channel_avg_like_rate、channel_name_len、channel_has_digit、channel_has_special\n")

# --------------------------
# 3. tags特征提取
# --------------------------
print("=== 2. 提取 tags 特征 ===")

# 3.1 标签密度（标签数/标题长度）
df_final['tag_density'] = np.where(
    df_final['title_length'] > 0,
    df_final['tags_count'] / df_final['title_length'],
    0
)  # 标签密度

# 3.2 热门标签占比：统计前10%热门标签，计算每个视频的热门标签比例
all_tags = []
for tag_str in df_final['tags']:
    if str(tag_str).strip() != '':
        all_tags.extend(str(tag_str).split('|'))
# 计算热门标签（出现频率前0.5%）
if len(all_tags) > 0:  # 避免无标签数据报错
    tag_freq = pd.Series(all_tags).value_counts()
    popular_threshold = int(len(tag_freq) * 0.001)  
    # 确保阈值至少为1（避免标签总数过少时threshold=0）
    popular_threshold = max(popular_threshold, 1)
    # 提取前10%热门标签（含标签名和出现次数）
    popular_tags_df = tag_freq.head(popular_threshold).reset_index()
    popular_tags_df.columns = ['热门标签名', '出现次数']
    popular_tags = set(popular_tags_df['热门标签名'])  # 用于后续计算占比
    
    # 打印前10%热门标签的名称和出现次数
    print(f"✅ 数据集中共有 {len(tag_freq)} 个标签，前0.1%热门标签（共 {popular_threshold} 个）如下：")
    print("-" * 50)
    print(popular_tags_df.to_string(index=False)) 
    print("-" * 50)
    
    # 计算热门标签占比
    def calc_popular_ratio(tag_str):
        if str(tag_str).strip() == '':
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

# 4.1 描述基础统计：描述长度、是否含实用信息（链接、时间戳）
df_final['desc_length'] = df_final['description'].apply(lambda x: len(str(x)) if pd.notna(x) and str(x).strip() != '' else 0)  # 描述长度（空描述设为0）
# 描述是否含 YouTube 链接（推荐其他视频，提升用户停留）
df_final['desc_has_youtube_link'] = df_final['description'].str.contains(r'youtube\.com|youtu\.be', regex=True, na=False).astype(int)
# 描述是否含时间戳（如“03:20 教程开始”，提升用户体验）
df_final['desc_has_timestamp'] = df_final['description'].str.contains(r'\d{1,2}:\d{2}', regex=True, na=False).astype(int)

# 4.2 描述关键词数量：统计视频相关关键词出现次数
key_words = ['tutorial', 'review', 'vlog', 'recipe', 'guide', 'tips', 'how to', 'best', 'top', 'learn', 'make', 'easy']
def count_keywords(desc):
    if pd.isna(desc) or str(desc).strip() == '':
        return 0
    desc_lower = str(desc).lower()
    return sum(1 for word in key_words if word in desc_lower)
df_final['desc_keyword_count'] = df_final['description'].apply(count_keywords)

# 删除原始描述列
df_final = df_final.drop(columns=['description'])
print(f"✅ 新增 4 个 description 特征：desc_length、desc_has_youtube_link、desc_has_timestamp、desc_keyword_count\n")


print("=== 文本特征提取完成总结 ===")
final_save_path = r"C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\USvideos_all_features.csv"
df_final.to_csv(final_save_path, index=False)
print(f"\n✅ 最终数据集已保存至：{final_save_path}")