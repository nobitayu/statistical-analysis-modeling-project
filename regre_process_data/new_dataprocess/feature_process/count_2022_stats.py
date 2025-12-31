# 脚本主要功能：
# 1. 数据统计：统计2022年发布的视频总数
# 2. 趋势统计：统计2022年发布且进入趋势榜（is_trending=1）的视频数量
# 3. 数据读取：支持多种编码格式（utf-8, latin1等）读取原始CSV文件

import pandas as pd
import os

# 定义文件路径
input_file_path = r'c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_all.csv'
output_file_path = r'c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Trending_Sampled.csv'

def main():
    if not os.path.exists(input_file_path):
        print(f"Error: File not found at {input_file_path}")
        return

    try:
        print("Starting data processing...")
        
        # 尝试不同的编码读取文件
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'gbk']
        df = None
        
        for enc in encodings:
            try:
                print(f"Trying encoding: {enc}")
                # 读取所有列
                df = pd.read_csv(input_file_path, encoding=enc)
                print(f"Successfully read with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading with {enc}: {e}")
                continue
        
        if df is None:
            raise Exception("Failed to read file with common encodings")
        
        # 转换 publishedAt 为 datetime
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
        
        # 筛选条件：publishedAt 在 2022 年 且 is_trending 为 1
        # 注意：is_trending 可能是数字或字符串，先转为数字处理
        df['is_trending_numeric'] = pd.to_numeric(df['is_trending'], errors='coerce')
        
        # 应用筛选
        filtered_df = df[
            (df['publishedAt'].dt.year == 2022) & 
            (df['is_trending_numeric'] == 1)
        ]
        
        print(f"Found {len(filtered_df)} rows matching criteria.")
        
        # 删除辅助列
        filtered_df = filtered_df.drop(columns=['is_trending_numeric'])
        
        # 随机抽取 5000 行
        sample_size = 5000
        if len(filtered_df) > sample_size:
            sampled_df = filtered_df.sample(n=sample_size, random_state=42) # 设置 random_state 保证可复现
            print(f"Successfully sampled {sample_size} rows.")
        else:
            sampled_df = filtered_df
            print(f"Rows count ({len(filtered_df)}) is less than or equal to {sample_size}, keeping all.")
            
        # 保存结果
        sampled_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"Saved sampled data to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
