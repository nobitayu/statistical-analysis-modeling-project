# 脚本主要功能：
# 1. 播放量分布可视化：绘制 view_count 的箱线图
# 2. 类别占比可视化：绘制 categoryId 的扇形图（映射为名称）
# 3. 标签分析：统计 tags 使用数量，列出前 1% 的热门标签
# 4. 标题特征分析：展示 title_upper_ratio 对 log_view_count 的影响（散点图/回归图）
# 5. 时间趋势分析：按发布时段（Hour）和周几（Weekday）统计平均观看量

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# 配置中文字体，避免绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 增加：调整全局字体大小 ===
plt.rcParams['font.size'] = 16          # 全局默认字体大小
plt.rcParams['axes.titlesize'] = 18     # 标题字体大小
plt.rcParams['axes.labelsize'] = 16     # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14    # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 14    # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 14    # 图例字体大小
# ==============================

# 文件路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Transformed.csv"
tags_info_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\tags info.txt"
output_dir = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\figure"

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=== 开始可视化分析 ===")
    
    # 1. 读取数据
    try:
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file_path, encoding='latin1')
        print(f"成功读取数据，形状：{df.shape}")
        
        # 使用硬编码的 category_dict
        category_dict = {
            1: 'Film & Animation',
            2: 'Autos & Vehicles',
            10: 'Music',
            15: 'Pets & Animals',
            17: 'Sports',
            18: 'Short Movies',
            19: 'Travel & Events',
            20: 'Gaming',
            21: 'Videoblogging',
            22: 'People & Blogs',
            23: 'Comedy',
            24: 'Entertainment',
            25: 'News & Politics',
            26: 'Howto & Style',
            27: 'Education',
            28: 'Science & Technology',
            29: 'Nonprofits & Activism'
        }

    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    # 预处理：转换时间格式
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # 确保 log_view_count 存在
    if 'log_view_count' not in df.columns:
         df['log_view_count'] = np.log(df['view_count'] + 1)

    # 确保 tags_count 存在 (如果不存在则计算)
    if 'tags_count' not in df.columns:
        if 'tags' in df.columns:
             # tags 分隔符通常为 '|'
             df['tags_count'] = df['tags'].apply(lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
        else:
             print("Warning: tags column missing, cannot calculate tags_count")
             df['tags_count'] = 0
    
    # --------------------------
    # 1. 对 log_view_count 画箱线图
    # --------------------------
    print("正在绘制 1. log_view_count 箱线图...")
    
    # 确保 log_view_count 存在
    if 'log_view_count' not in df.columns:
         df['log_view_count'] = np.log(df['view_count'] + 1)

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['log_view_count'])
    plt.title('Distribution of Log View Count (Boxplot)')
    plt.ylabel('Log View Count')
    # plt.yscale('log') # 已是对数变换后的数据，无需对数坐标
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, '1_log_view_count_boxplot.png'))
    plt.close()

    # --------------------------
    # 2. categoryId 画扇形图
    # --------------------------
    print("正在绘制 2. Category 扇形图...")
    if 'categoryId' in df.columns:
        # 统计每个类别的数量
        cat_counts = df['categoryId'].value_counts()
        
        # 映射名称
        labels = [category_dict.get(cat_id, f"ID: {cat_id}") for cat_id in cat_counts.index]
        
        plt.figure(figsize=(12, 12))
        # 仅展示前 15 个大类，其他的归为 'Others' 以免饼图太乱
        top_n = 15
        if len(cat_counts) > top_n:
            top_counts = cat_counts[:top_n]
            others_count = cat_counts[top_n:].sum()
            top_labels = labels[:top_n]
            
            # 添加 Others
            data_to_plot = list(top_counts.values) + [others_count]
            labels_to_plot = top_labels + ['Others']
        else:
            data_to_plot = cat_counts.values
            labels_to_plot = labels
            
        plt.pie(data_to_plot, labels=labels_to_plot, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
        plt.title(f'Category Distribution (Top {top_n})')
        plt.axis('equal') # 保证饼图是圆的
        plt.savefig(os.path.join(output_dir, '2_category_pie_chart.png'))
        plt.close()
    else:
        print("Warning: categoryId 列不存在，跳过扇形图绘制。")

    # --------------------------
    # 3. 统计 tags 使用数量，列出前 1% 的热门 tag
    # --------------------------
    print("正在统计 3. 热门 Tags...")
    if 'tags' in df.columns:
        all_tags = []
        # tags 是用 | 分隔的字符串
        for tags_str in df['tags'].dropna():
            if isinstance(tags_str, str):
                # 分割并去除两端空格
                tags_list = [t.strip() for t in tags_str.split('|') if t.strip()]
                all_tags.extend(tags_list)
        
        tag_counts = Counter(all_tags)
        total_unique_tags = len(tag_counts)
        top_1_percent_count = max(1, int(total_unique_tags * 0.01))
        
        most_common_tags = tag_counts.most_common(top_1_percent_count)
        
        # 保存到 txt
        with open(os.path.join(output_dir, '3_top_1_percent_tags.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Total Unique Tags: {total_unique_tags}\n")
            f.write(f"Top 1% Count: {top_1_percent_count}\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Rank':<5} | {'Tag':<50} | {'Count':<10}\n")
            f.write("-" * 30 + "\n")
            for i, (tag, count) in enumerate(most_common_tags):
                f.write(f"{i+1:<5} | {tag:<50} | {count:<10}\n")
                
        # 顺便画个词云或柱状图展示前20个
        plt.figure(figsize=(12, 8))
        top_20_tags = tag_counts.most_common(20)
        tags_x = [x[0] for x in top_20_tags]
        tags_y = [x[1] for x in top_20_tags]
        sns.barplot(x=tags_y, y=tags_x, palette='viridis')
        plt.title('Top 20 Most Frequent Tags')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_top_20_tags_bar.png'))
        plt.close()

    # --------------------------
    # 4. 展示 title_upper_ratio 对 log_view_count 的影响
    # --------------------------
    print("正在绘制 4. Title Upper Ratio vs Log View Count...")
    # 确保 log_view_count 存在，如果不存在则计算
    if 'log_view_count' not in df.columns:
         df['log_view_count'] = np.log(df['view_count'] + 1)
         
    plt.figure(figsize=(10, 6))
    # 使用 Hexbin 图或散点图 (数据量较大时 hexbin 更好看密度)
    # 这里用散点图 + 回归线
    sns.regplot(x='title_upper_ratio', y='log_view_count', data=df, 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.title('Impact of Title Uppercase Ratio on Views (Log Scale)')
    plt.xlabel('Title Uppercase Ratio')
    plt.ylabel('Log(View Count)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, '4_upper_ratio_vs_views.png'))
    plt.close()

    # --------------------------
    # 5. 按发布时间段/周几统计平均观看量
    # --------------------------
    print("正在绘制 5. 时间段/周几 平均播放量统计...")
    
    # 提取 Hour 和 Weekday
    df['hour'] = df['publishedAt'].dt.hour
    df['weekday'] = df['publishedAt'].dt.weekday # 0=Monday, 6=Sunday
    
    # 映射 Weekday 名称
    weekday_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    df['weekday_name'] = df['weekday'].map(weekday_map)
    
    # 计算聚合数据
    hourly_avg = df.groupby('hour')['view_count'].mean().reset_index()
    weekday_avg = df.groupby('weekday')['view_count'].mean().reset_index()
    weekday_avg['weekday_name'] = weekday_avg['weekday'].map(weekday_map)

    # 绘图：双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 子图1：一天内不同小时的平均播放量
    sns.lineplot(x='hour', y='view_count', data=hourly_avg, marker='o', ax=ax1, color='b')
    ax1.set_title('Average View Count by Publish Hour')
    ax1.set_xlabel('Hour of Day (0-23)')
    ax1.set_ylabel('Average View Count')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True)
    
    # 子图2：一周内不同日子的平均播放量
    sns.barplot(x='weekday_name', y='view_count', data=weekday_avg, ax=ax2, palette='coolwarm')
    ax2.set_title('Average View Count by Day of Week')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average View Count')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_time_analysis_views.png'))
    plt.close()

    # --------------------------
    # 6. popular_tag_ratio 对 log_view_count 的影响
    # --------------------------
    print("正在绘制 6. Popular Tag Ratio vs Log View Count...")
    if 'popular_tag_ratio' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='popular_tag_ratio', y='log_view_count', data=df, 
                    scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
        plt.title('Impact of Popular Tag Ratio on Log Views')
        plt.xlabel('Popular Tag Ratio')
        plt.ylabel('Log(View Count)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(output_dir, '6_popular_tag_ratio_vs_views.png'))
        plt.close()
    else:
        print("Warning: popular_tag_ratio column missing.")

    # --------------------------
    # 7. tags_count 对 log_view_count 的影响
    # --------------------------
    print("正在绘制 7. Tags Count vs Log View Count...")
    plt.figure(figsize=(10, 6))
    # 对于 tags_count，由于是离散值，regplot 依然适用，但也可以考虑 boxplot
    # 这里使用 regplot 展示趋势，x_jitter 可以帮助分散重叠的点
    sns.regplot(x='tags_count', y='log_view_count', data=df, x_jitter=0.2,
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'green'})
    plt.title('Impact of Number of Tags on Log Views')
    plt.xlabel('Number of Tags')
    plt.ylabel('Log(View Count)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, '7_tags_count_vs_views.png'))
    plt.close()

    print("\n所有图表绘制完成！保存在:", output_dir)

if __name__ == "__main__":
    main()
