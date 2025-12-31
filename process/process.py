import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
print("正在读取数据...")
try:
    df = pd.read_csv('Youtube_Videos.csv')
    print(f"数据读取成功，总行数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    
except FileNotFoundError:
    print("错误: 找不到Youtube_Videos.csv文件")
    exit()

# 2. 数据清洗和筛选
print("\n" + "="*50)
print("数据清洗和筛选")
print("="*50)

# 2.1 处理日期时间
print("\n1. 处理日期时间...")

def parse_dates(date_series):
    """智能解析日期时间"""
    # 尝试多种格式
    for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
        try:
            parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except:
            continue
    
    # 最后尝试自动推断
    return pd.to_datetime(date_series, errors='coerce')

df['publishedAt'] = parse_dates(df['publishedAt'])
df['trending_date'] = parse_dates(df['trending_date'])

print(f"   publishedAt - 有效值: {df['publishedAt'].notna().sum()}/{len(df)}")
print(f"   trending_date - 有效值: {df['trending_date'].notna().sum()}/{len(df)}")

# 2.2 筛选2022年的数据
print("\n2. 筛选2022年数据...")
if df['publishedAt'].notna().sum() > 0:
    df_2022 = df[df['publishedAt'].dt.year == 2022].copy()
    print(f"   2022年数据行数: {len(df_2022)}")
else:
    print("   警告: 没有有效的publishedAt日期，使用全部数据")
    df_2022 = df.copy()

# 2.3 处理缺失值
print("\n3. 处理缺失值...")
print(f"   处理前行数: {len(df_2022)}")

# 确定关键列
key_columns = ['video_id', 'title', 'publishedAt', 'channelId', 'channelTitle', 
               'categoryId', 'view_count', 'likes', 'comment_count', 'is_trending']

# 只删除关键列有缺失值的行
initial_rows = len(df_2022)
df_clean = df_2022.dropna(subset=key_columns)
print(f"   删除关键列缺失值后行数: {len(df_clean)} (删除了 {initial_rows - len(df_clean)} 行)")

# 2.4 检查is_trending列的值
print("\n4. 检查is_trending列...")
trending_counts = df_clean['is_trending'].value_counts()
print(f"   is_trending值分布:")
for value, count in trending_counts.items():
    print(f"     {value}: {count}行 ({count/len(df_clean):.1%})")

# 2.5 抽样 - 确保热门和非热门各2500条
print("\n5. 抽样...")
TARGET_EACH = 2500  # 每类2500条
TARGET_TOTAL = TARGET_EACH * 2  # 总共5000条

# 计算各类别的可用数量
trending_available = trending_counts.get(1, 0)
non_trending_available = trending_counts.get(0, 0)

print(f"   可用数据:")
print(f"     热门视频(is_trending=1): {trending_available}行")
print(f"     非热门视频(is_trending=0): {non_trending_available}行")

# 检查是否有足够的数据
if trending_available < TARGET_EACH:
    print(f"   ⚠️ 警告: 热门视频不足{TARGET_EACH}条，只有{trending_available}条")
    print(f"   将抽取所有{trending_available}条热门视频")
    trending_sample_size = trending_available
else:
    trending_sample_size = TARGET_EACH

if non_trending_available < TARGET_EACH:
    print(f"   ⚠️ 警告: 非热门视频不足{TARGET_EACH}条，只有{non_trending_available}条")
    print(f"   将抽取所有{non_trending_available}条非热门视频")
    non_trending_sample_size = non_trending_available
else:
    non_trending_sample_size = TARGET_EACH

print(f"\n   计划抽样:")
print(f"     热门视频: {trending_sample_size}行")
print(f"     非热门视频: {non_trending_sample_size}行")
print(f"     总计: {trending_sample_size + non_trending_sample_size}行")

# 进行分层抽样
sample_dfs = []

# 抽取热门视频
if trending_sample_size > 0:
    if trending_available >= trending_sample_size:
        trending_sample = df_clean[df_clean['is_trending'] == 1].sample(
            trending_sample_size, random_state=42, replace=False
        )
        sample_dfs.append(trending_sample)
        print(f"   ✅ 成功抽取 {trending_sample_size} 条热门视频")
    else:
        print(f"   ⚠️ 警告: 热门视频数量不足，无法抽取 {trending_sample_size} 条")

# 抽取非热门视频
if non_trending_sample_size > 0:
    if non_trending_available >= non_trending_sample_size:
        non_trending_sample = df_clean[df_clean['is_trending'] == 0].sample(
            non_trending_sample_size, random_state=42, replace=False
        )
        sample_dfs.append(non_trending_sample)
        print(f"   ✅ 成功抽取 {non_trending_sample_size} 条非热门视频")
    else:
        print(f"   ⚠️ 警告: 非热门视频数量不足，无法抽取 {non_trending_sample_size} 条")

# 合并样本
if sample_dfs:
    sample_df = pd.concat(sample_dfs, ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序
    
    print(f"\n   ✅ 最终抽取 {len(sample_df)} 行数据")
else:
    print("   ❌ 错误: 无法抽取任何数据")
    exit()

print(f"\n最终样本大小: {len(sample_df)}行")
trending_final = sample_df['is_trending'].value_counts()
print("样本中is_trending分布:")
for value, count in trending_final.items():
    label = "热门" if value == 1 else "非热门"
    print(f"  {label}视频(is_trending={value}): {count}行 ({count/len(sample_df):.1%})")

# 3. 特征工程
print("\n" + "="*50)
print("特征工程")
print("="*50)

# 3.1 标题长度
sample_df['title_length'] = sample_df['title'].astype(str).str.len()

# 3.2 问号感叹号数量
sample_df['question_exclamation_count'] = sample_df['title'].astype(str).apply(
    lambda x: x.count('?') + x.count('!')
)

# 3.3 标签数量
def count_tags(tag_str):
    if pd.isna(tag_str):
        return 0
    tag_str = str(tag_str)
    if tag_str.lower() == 'nan' or tag_str.strip() == '':
        return 0
    # 分割标签
    tags = [tag.strip() for tag in tag_str.split(',') if tag.strip()]
    return len(tags)

sample_df['tag_count'] = sample_df['tags'].apply(count_tags)

# 3.4 标签是否出现在标题里
def tags_in_title(row):
    if pd.isna(row['tags']) or str(row['tags']).lower() == 'nan':
        return 0
    
    title = str(row['title']).lower()
    tags = str(row['tags']).lower()
    
    if not tags.strip():
        return 0
    
    # 分割标签
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    for tag in tag_list:
        if tag and tag in title:
            return 1
    return 0

sample_df['tags_in_title'] = sample_df.apply(tags_in_title, axis=1)

# 3.5 发布时间是否周末
if pd.api.types.is_datetime64_any_dtype(sample_df['publishedAt']):
    sample_df['is_weekend'] = sample_df['publishedAt'].dt.dayofweek.isin([5, 6]).astype(int)
else:
    print("警告: publishedAt不是datetime类型，无法确定是否周末")
    sample_df['is_weekend'] = 0

# 3.6 发布一天的具体时间
def get_time_period(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'dawn'

if pd.api.types.is_datetime64_any_dtype(sample_df['publishedAt']):
    sample_df['hour'] = sample_df['publishedAt'].dt.hour
    sample_df['time_period'] = sample_df['hour'].apply(get_time_period)
else:
    print("警告: publishedAt不是datetime类型，无法确定发布时间段")
    sample_df['hour'] = 12
    sample_df['time_period'] = 'afternoon'

# 3.7 类别映射
category_mapping = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    19: 'Travel & Events',
    20: 'Gaming',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism'
}

sample_df['category'] = sample_df['categoryId'].map(category_mapping)

# 处理未映射的类别
if sample_df['category'].isna().any():
    unknown_categories = sample_df[sample_df['category'].isna()]['categoryId'].unique()
    print(f"警告: 发现未映射的类别ID: {unknown_categories}")
    sample_df['category'] = sample_df['category'].fillna('Unknown')

print("特征工程完成!")
print("\n新增特征统计摘要:")
new_features = ['title_length', 'question_exclamation_count', 'tag_count', 
                'tags_in_title', 'is_weekend', 'time_period', 'category']
print(sample_df[new_features].describe(include='all'))

# 4. 可视化分析
print("\n" + "="*50)
print("可视化分析")
print("="*50)

# 设置图形大小和布局
fig = plt.figure(figsize=(20, 18))
fig.suptitle('YouTube视频特征分析 (热门 vs 非热门各2500条)', fontsize=18, fontweight='bold', y=1.02)

# 4.1 标题长度 vs 是否热门 (箱线图)
ax1 = plt.subplot(3, 3, 1)
sns.boxplot(x='is_trending', y='title_length', data=sample_df, ax=ax1)
ax1.set_title('标题长度 vs 是否热门', fontsize=14)
ax1.set_xlabel('是否热门 (0=非热门, 1=热门)', fontsize=12)
ax1.set_ylabel('标题长度', fontsize=12)
# 添加均值标记
trending_mean = sample_df[sample_df['is_trending'] == 1]['title_length'].mean()
non_trending_mean = sample_df[sample_df['is_trending'] == 0]['title_length'].mean()
ax1.text(0, non_trending_mean, f'均值: {non_trending_mean:.1f}', ha='center', va='bottom', fontsize=10)
ax1.text(1, trending_mean, f'均值: {trending_mean:.1f}', ha='center', va='bottom', fontsize=10)

# 4.2 问号感叹号数量 vs 是否热门 (箱线图)
ax2 = plt.subplot(3, 3, 2)
sns.boxplot(x='is_trending', y='question_exclamation_count', data=sample_df, ax=ax2)
ax2.set_title('问号感叹号数量 vs 是否热门', fontsize=14)
ax2.set_xlabel('是否热门 (0=非热门, 1=热门)', fontsize=12)
ax2.set_ylabel('问号感叹号数量', fontsize=12)
# 添加均值标记
trending_qe_mean = sample_df[sample_df['is_trending'] == 1]['question_exclamation_count'].mean()
non_trending_qe_mean = sample_df[sample_df['is_trending'] == 0]['question_exclamation_count'].mean()
ax2.text(0, non_trending_qe_mean, f'均值: {non_trending_qe_mean:.2f}', ha='center', va='bottom', fontsize=10)
ax2.text(1, trending_qe_mean, f'均值: {trending_qe_mean:.2f}', ha='center', va='bottom', fontsize=10)

# 4.3 标签数量 vs 是否热门 (箱线图)
ax3 = plt.subplot(3, 3, 3)
sns.boxplot(x='is_trending', y='tag_count', data=sample_df, ax=ax3)
ax3.set_title('标签数量 vs 是否热门', fontsize=14)
ax3.set_xlabel('是否热门 (0=非热门, 1=热门)', fontsize=12)
ax3.set_ylabel('标签数量', fontsize=12)
# 添加均值标记
trending_tags_mean = sample_df[sample_df['is_trending'] == 1]['tag_count'].mean()
non_trending_tags_mean = sample_df[sample_df['is_trending'] == 0]['tag_count'].mean()
ax3.text(0, non_trending_tags_mean, f'均值: {non_trending_tags_mean:.2f}', ha='center', va='bottom', fontsize=10)
ax3.text(1, trending_tags_mean, f'均值: {trending_tags_mean:.2f}', ha='center', va='bottom', fontsize=10)

# 4.4 标签是否出现在标题里 vs 是否热门 (条形图)
ax4 = plt.subplot(3, 3, 4)
tags_counts = sample_df.groupby(['tags_in_title', 'is_trending']).size().unstack()
tags_counts.plot(kind='bar', ax=ax4, color=['#1f77b4', '#ff7f0e'])
ax4.set_title('标签是否出现在标题里 vs 是否热门', fontsize=14)
ax4.set_xlabel('标签在标题中 (0=否, 1=是)', fontsize=12)
ax4.set_ylabel('视频数量', fontsize=12)
ax4.legend(['非热门', '热门'], fontsize=10)
# 添加数值标签
for i, (idx, row) in enumerate(tags_counts.iterrows()):
    for j, value in enumerate(row):
        if not pd.isna(value):
            ax4.text(i + (-0.2 + j*0.4), value + max(tags_counts.max().max()*0.01, 5), 
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

# 4.5 发布时间是否周末 vs 是否热门 (条形图)
ax5 = plt.subplot(3, 3, 5)
weekend_counts = sample_df.groupby(['is_weekend', 'is_trending']).size().unstack()
weekend_counts.plot(kind='bar', ax=ax5, color=['#1f77b4', '#ff7f0e'])
ax5.set_title('发布时间是否周末 vs 是否热门', fontsize=14)
ax5.set_xlabel('是否周末 (0=工作日, 1=周末)', fontsize=12)
ax5.set_ylabel('视频数量', fontsize=12)
ax5.legend(['非热门', '热门'], fontsize=10)
# 添加数值标签
for i, (idx, row) in enumerate(weekend_counts.iterrows()):
    for j, value in enumerate(row):
        if not pd.isna(value):
            ax5.text(i + (-0.2 + j*0.4), value + max(weekend_counts.max().max()*0.01, 5), 
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

# 4.6 发布时间段 vs 是否热门 (条形图)
ax6 = plt.subplot(3, 3, 6)
time_counts = sample_df.groupby(['time_period', 'is_trending']).size().unstack()
time_counts = time_counts.reindex(['dawn', 'morning', 'afternoon', 'evening'])  # 按时间顺序排列
time_counts.plot(kind='bar', ax=ax6, color=['#1f77b4', '#ff7f0e'])
ax6.set_title('发布时间段 vs 是否热门', fontsize=14)
ax6.set_xlabel('时间段', fontsize=12)
ax6.set_ylabel('视频数量', fontsize=12)
ax6.legend(['非热门', '热门'], fontsize=10)
# 添加数值标签
for i, (idx, row) in enumerate(time_counts.iterrows()):
    for j, value in enumerate(row):
        if not pd.isna(value):
            ax6.text(i + (-0.2 + j*0.4), value + max(time_counts.max().max()*0.01, 5), 
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)

# 4.7 类别 vs 是否热门 (条形图，前10个类别)
ax7 = plt.subplot(3, 3, 7)
top_categories = sample_df['category'].value_counts().head(10).index
category_data = sample_df[sample_df['category'].isin(top_categories)]
cat_counts = category_data.groupby(['category', 'is_trending']).size().unstack()
cat_counts.plot(kind='bar', ax=ax7, color=['#1f77b4', '#ff7f0e'])
ax7.set_title('类别 vs 是否热门 (前10类别)', fontsize=14)
ax7.set_xlabel('类别', fontsize=12)
ax7.set_ylabel('视频数量', fontsize=12)
ax7.legend(['非热门', '热门'], fontsize=10)
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
# 添加数值标签
for i, (idx, row) in enumerate(cat_counts.iterrows()):
    for j, value in enumerate(row):
        if not pd.isna(value):
            ax7.text(i + (-0.2 + j*0.4), value + max(cat_counts.max().max()*0.01, 5), 
                    f'{int(value)}', ha='center', va='bottom', fontsize=8)

# 4.8 标题长度 vs 标签数量散点图 (按是否热门着色)
ax8 = plt.subplot(3, 3, 8)
scatter = ax8.scatter(sample_df['title_length'], 
                     sample_df['tag_count'], 
                     c=sample_df['is_trending'], 
                     alpha=0.6, 
                     cmap='coolwarm',
                     s=30)
ax8.set_title('标题长度 vs 标签数量', fontsize=14)
ax8.set_xlabel('标题长度', fontsize=12)
ax8.set_ylabel('标签数量', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('是否热门 (0=非热门, 1=热门)', fontsize=10)

# 4.9 发布时间小时分布 (直方图)
ax9 = plt.subplot(3, 3, 9)
if 'hour' in sample_df.columns:
    trending_hours = sample_df[sample_df['is_trending'] == 1]['hour']
    non_trending_hours = sample_df[sample_df['is_trending'] == 0]['hour']
    
    ax9.hist([trending_hours, non_trending_hours], 
            bins=24, 
            alpha=0.7, 
            label=['热门', '非热门'],
            color=['#ff7f0e', '#1f77b4'])
    ax9.set_title('发布时间小时分布 vs 是否热门', fontsize=14)
    ax9.set_xlabel('小时 (0-23)', fontsize=12)
    ax9.set_ylabel('视频数量', fontsize=12)
    ax9.legend(fontsize=10)
    ax9.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('youtube_analysis_balanced_5000.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 统计检验
print("\n" + "="*50)
print("统计检验结果")
print("="*50)

def perform_ttest(data, var_name, group_var='is_trending'):
    """执行T检验的辅助函数"""
    group1 = data[data[group_var] == 1][var_name]
    group0 = data[data[group_var] == 0][var_name]
    
    if len(group1) < 2 or len(group0) < 2:
        print(f"{var_name}: 数据不足，无法进行T检验")
        return None, None
    
    t_stat, p_value = stats.ttest_ind(group1, group0, equal_var=False)
    return t_stat, p_value

# 5.1 标题长度
print("\n1. 标题长度 vs 是否热门")
t_stat, p_value = perform_ttest(sample_df, 'title_length')
if t_stat is not None:
    trending_len = sample_df[sample_df['is_trending'] == 1]['title_length'].mean()
    non_trending_len = sample_df[sample_df['is_trending'] == 0]['title_length'].mean()
    
    print(f"   H₀: 热门和非热门视频的标题长度无显著差异")
    print(f"   热门视频平均标题长度: {trending_len:.2f}")
    print(f"   非热门视频平均标题长度: {non_trending_len:.2f}")
    print(f"   差异: {trending_len - non_trending_len:.2f}")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   P值: {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，标题长度与是否热门有显著差异 (p < 0.05)")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，标题长度与是否热门无显著差异 (p = {p_value:.4f})")

# 5.2 问号感叹号数量
print("\n2. 问号感叹号数量 vs 是否热门")
t_stat, p_value = perform_ttest(sample_df, 'question_exclamation_count')
if t_stat is not None:
    trending_qe = sample_df[sample_df['is_trending'] == 1]['question_exclamation_count'].mean()
    non_trending_qe = sample_df[sample_df['is_trending'] == 0]['question_exclamation_count'].mean()
    
    print(f"   H₀: 热门和非热门视频的问号感叹号数量无显著差异")
    print(f"   热门视频平均数量: {trending_qe:.2f}")
    print(f"   非热门视频平均数量: {non_trending_qe:.2f}")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，问号感叹号数量与是否热门有显著差异")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，问号感叹号数量与是否热门无显著差异")

# 5.3 标签数量
print("\n3. 标签数量 vs 是否热门")
t_stat, p_value = perform_ttest(sample_df, 'tag_count')
if t_stat is not None:
    trending_tags = sample_df[sample_df['is_trending'] == 1]['tag_count'].mean()
    non_trending_tags = sample_df[sample_df['is_trending'] == 0]['tag_count'].mean()
    
    print(f"   H₀: 热门和非热门视频的标签数量无显著差异")
    print(f"   热门视频平均标签数: {trending_tags:.2f}")
    print(f"   非热门视频平均标签数: {non_trending_tags:.2f}")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   P值: {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，标签数量与是否热门有显著差异")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，标签数量与是否热门无显著差异")

# 5.4 标签是否出现在标题里 (卡方检验)
print("\n4. 标签是否出现在标题里 vs 是否热门")
contingency_table = pd.crosstab(sample_df['tags_in_title'], sample_df['is_trending'])
print(f"   列联表:")
print(contingency_table)

if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"   H₀: 标签是否出现在标题里与是否热门无关")
    print(f"   卡方统计量: {chi2:.4f}")
    print(f"   P值: {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，标签是否出现在标题里与是否热门有关")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，标签是否出现在标题里与是否热门无关")

# 5.5 发布时间是否周末 (卡方检验)
print("\n5. 发布时间是否周末 vs 是否热门")
print(f"   H₀: 周末发布的视频是否为热门视频与工作日无显著差异")
contingency_table = pd.crosstab(sample_df['is_weekend'], sample_df['is_trending'])
print(f"   列联表:")
print(contingency_table)

if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"   卡方统计量: {chi2:.4f}")
    print(f"   P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，周末发布的视频是否为热门视频与工作日有显著差异")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，周末发布的视频是否为热门视频与工作日无显著差异")

# 5.6 发布时间段 (卡方检验)
print("\n6. 发布时间段 vs 是否热门")
contingency_table = pd.crosstab(sample_df['time_period'], sample_df['is_trending'])
print(f"   列联表:")
print(contingency_table)

if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"   H₀: 不同时间段发布的视频是否为热门视频无显著差异")
    print(f"   卡方统计量: {chi2:.4f}")
    print(f"   P值: {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，不同时间段发布的视频是否为热门视频有显著差异")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，不同时间段发布的视频是否为热门视频无显著差异")

# 5.7 类别 (卡方检验)
print("\n7. 类别 vs 是否热门")
contingency_table = pd.crosstab(sample_df['category'], sample_df['is_trending'])
print(f"   H₀: 不同类别的视频是否为热门视频无显著差异")

if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"   卡方统计量: {chi2:.4f}")
    print(f"   P值: {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"   ✅ 结论: 拒绝H₀，不同类别的视频是否为热门视频有显著差异")
    else:
        print(f"   ❌ 结论: 不能拒绝H₀，不同类别的视频是否为热门视频无显著差异")

# 保存处理后的数据
output_file = 'youtube_data_balanced_5000.csv'
sample_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n处理后的数据已保存为: {output_file}")
print(f"文件大小: {len(sample_df)} 行 × {len(sample_df.columns)} 列")

# 数据摘要
print("\n" + "="*50)
print("数据摘要")
print("="*50)
print(f"总样本数: {len(sample_df)}")
print(f"热门视频: {sample_df['is_trending'].sum()} ({sample_df['is_trending'].mean()*100:.1f}%)")
print(f"非热门视频: {len(sample_df) - sample_df['is_trending'].sum()} ({(1-sample_df['is_trending'].mean())*100:.1f}%)")

if pd.api.types.is_datetime64_any_dtype(sample_df['publishedAt']):
    print(f"样本时间段: {sample_df['publishedAt'].min().date()} 到 {sample_df['publishedAt'].max().date()}")

print("\n各特征统计摘要:")
print("1. 标题长度:")
print(f"   最小值: {sample_df['title_length'].min()}, 最大值: {sample_df['title_length'].max()}, 平均值: {sample_df['title_length'].mean():.2f}")

print("\n2. 标签数量:")
print(f"   最小值: {sample_df['tag_count'].min()}, 最大值: {sample_df['tag_count'].max()}, 平均值: {sample_df['tag_count'].mean():.2f}")

print("\n3. 周末发布比例:")
weekend_percent = sample_df['is_weekend'].mean() * 100
print(f"   周末发布的视频: {weekend_percent:.1f}%")

print("\n4. 时间段分布:")
time_dist = sample_df['time_period'].value_counts(normalize=True) * 100
for period, percent in time_dist.items():
    print(f"   {period}: {percent:.1f}%")

print("\n5. 类别分布 (前5):")
category_dist = sample_df['category'].value_counts(normalize=True).head(5) * 100
for category, percent in category_dist.items():
    print(f"   {category}: {percent:.1f}%")

print("\n分析完成! ✓")