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
