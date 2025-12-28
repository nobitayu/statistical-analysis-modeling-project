# 1_preprocess.py
import pandas as pd
import numpy as np
import os
import config

def preprocess():
    print(">>> [Step 1] 开始数据预处理...")
    
    # 1. 读取数据
    if not os.path.exists(config.DATA_PATH):
        print(f"错误：找不到文件 {config.DATA_PATH}")
        return
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"原始数据形状: {df.shape}")

    # 2. 目标变量对数化 (log(x+1) 避免 0 值报错)
    # 生成新列 log_views
    df['log_views'] = np.log1p(df[config.TARGET_COL])
    print(f"已生成目标变量: log_views (对 {config.TARGET_COL} 进行 Log 变换)")

    # 3. 提取需要的特征列
    # 动态获取所有的 category_xxx 和 period_xxx 列
    cat_cols = [c for c in df.columns if c.startswith(config.PREFIX_CATEGORY)]
    period_cols = [c for c in df.columns if c.startswith(config.PREFIX_PERIOD)]
    
    print(f"检测到分类变量 (Category): {len(cat_cols)} 个")
    print(f"检测到时段变量 (Period): {len(period_cols)} 个")

    # ================= 核心：哑变量处理 (Dummy Variable Trap) =================
    # 为了避免完全共线性，必须每组扔掉一个。这里我们扔掉列表中的最后一个。
    # 比如 period 有 4 个，我们只保留前 3 个。如果全为 0，就代表是第 4 个。
    if len(cat_cols) > 1:
        drop_cat = cat_cols[-1]
        cat_cols = cat_cols[:-1]
        print(f"  -> 为了统计建模，剔除基准分类列: {drop_cat}")
    
    if len(period_cols) > 1:
        drop_period = period_cols[-1]
        period_cols = period_cols[:-1]
        print(f"  -> 为了统计建模，剔除基准时段列: {drop_period}")

    # 4. 组合最终的 X 特征
    final_features = config.NUMERIC_COLS + config.BINARY_COLS + cat_cols + period_cols
    
    # 检查这些列是否存在，防止报错
    available_cols = [c for c in final_features if c in df.columns]
    
    # 创建只包含有用数据的 DataFrame
    # 必须包含 log_views 用于后续训练
    df_clean = df[available_cols + ['log_views']].copy()

    # 5. 填充缺失值 (用 0 填充，或者你可以改成用 mean 填充)
    df_clean.fillna(0, inplace=True)

    # 6. 保存
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(config.PROCESSED_DATA_PATH, index=False)
    
    print("-" * 30)
    print(f"处理完成！最终特征数量: {len(available_cols)}")
    print(f"清洗后的数据已保存至: {config.PROCESSED_DATA_PATH}")
    print("请继续运行 step 2。")

if __name__ == "__main__":
    preprocess()