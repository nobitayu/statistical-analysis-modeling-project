# 脚本主要功能：
# 1. API调用：使用YouTube Data API获取美国地区（US）的视频分类数据
# 2. 数据映射：建立分类ID（category_id）到分类名称（category_name）的映射关系
# 3. 结果保存：将映射表保存为CSV文件，用于后续数据分析中的类别解析

import requests
import pandas as pd

API_KEY = "AIzaSyDaZICrDbp11KWtMC1ZMSL62iQ1XIG3q_0"
REGION_CODE = "US"  # 常用 US，Kaggle trending 数据基本都用 US

url = "https://www.googleapis.com/youtube/v3/videoCategories"

params = {
    "part": "snippet",
    "regionCode": REGION_CODE,
    "key": API_KEY
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# 解析 category_id -> category_name
categories = []
for item in data["items"]:
    categories.append({
        "category_id": int(item["id"]),
        "category_name": item["snippet"]["title"]
    })

df_categories = pd.DataFrame(categories).sort_values("category_id")

print(df_categories)

# 保存为 CSV（推荐）
df_categories.to_csv("youtube_category_mapping.csv", index=False)
