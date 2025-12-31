# 脚本主要功能：
# 1. 数据读取：读取 New_Youtube_Videos_2022_Transformed.csv
# 2. 数据预处理：排除前8列（A-H）原始数据，处理布尔值和缺失值
# 3. VIF计算：计算剩余特征的方差膨胀因子（VIF），评估多重共线性
# 4. 结果输出：打印并保存各特征的 VIF 值

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# 定义文件路径
input_file_path = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\New_Youtube_Videos_2022_Transformed.csv"
output_dir = r"c:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\new_dataprocess\dataset\analysis_output"
output_file_path = os.path.join(output_dir, "vif_results.csv")

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    print("=== 开始 VIF 分析 ===")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file_path, encoding='latin1')
    
    print(f"原始数据形状: {df.shape}")

    # 2. 排除前9列 (A-H)
    # A: title, B: publishedAt, C: categoryId, D: trending_date, E: tags, F: view_count, G: likes, H: comment_count, I: log_view_count(因变量)
    # 也可以直接通过列索引排除
    df_features = df.iloc[:, 9:].copy()
    print(f"排除前9列后数据形状: {df_features.shape}")
    print("保留的列名示例:", df_features.columns[:5].tolist())

    # 3. 数据预处理
    # 3.1 处理布尔值 (TRUE/FALSE -> 1/0)
    # 自动识别 object/bool 类型的列并转换
    for col in df_features.columns:
        if df_features[col].dtype == 'object' or df_features[col].dtype == 'bool':
            # 尝试转换为数值，'TRUE'/'FALSE' 会变成 NaN (如果直接用to_numeric)，或者我们需要手动映射
            # 先检查是否包含布尔字符串
            unique_vals = df_features[col].dropna().unique()
            if set(unique_vals).issubset({True, False, 'TRUE', 'FALSE', 'True', 'False', 0, 1, 0.0, 1.0}):
                df_features[col] = df_features[col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})
                # 填充剩余的可能是数值的
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    
    # 3.2 确保所有列都是数值型
    df_features = df_features.apply(pd.to_numeric, errors='coerce')

    # 3.3 处理无穷值和缺失值
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    null_counts = df_features.isnull().sum()
    if null_counts.sum() > 0:
        print(f"发现缺失值/无穷值，正在删除... (共 {null_counts.sum()} 个单元格)")
        df_features.dropna(inplace=True)
    
    print(f"用于 VIF 计算的数据形状: {df_features.shape}")

    if df_features.shape[0] == 0:
        print("错误：数据预处理后没有剩余行，无法计算 VIF。")
        return

    # 3.4 添加常数项 (截距)，用于 VIF 计算
    # 注意：如果数据已经包含常数项或完全共线，add_constant 会处理
    # 只有当数据中没有常数项时才需要添加，VIF 计算通常需要截距
    df_vif_input = add_constant(df_features)

    # 4. 计算 VIF
    print("正在计算 VIF (这可能需要一些时间)...")
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_vif_input.columns
    
    # 列表推导式计算每个特征的 VIF
    vif_data["VIF"] = [variance_inflation_factor(df_vif_input.values, i) 
                       for i in range(df_vif_input.shape[1])]
    
    # 排除常数项的显示 (通常不需要看 const 的 VIF)
    vif_data = vif_data[vif_data["feature"] != "const"]
    
    # 按 VIF 降序排列
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    # 5. 输出结果
    print("\n=== VIF 分析结果 (Top 20) ===")
    print(vif_data.head(20))
    
    # 保存结果
    vif_data.to_csv(output_file_path, index=False)
    print(f"\n完整 VIF 结果已保存至: {output_file_path}")

    # 简单解读
    print("\n[分析建议]")
    print("- VIF < 5: 多重共线性不严重")
    print("- 5 < VIF < 10: 存在中度多重共线性，需关注")
    print("- VIF > 10: 存在严重多重共线性，建议移除或合并相关特征")
    
    # 检查是否有无穷大
    if np.isinf(vif_data['VIF']).any():
        print("警告：存在 VIF 为无穷大 (inf) 的特征，这通常意味着存在完全多重共线性（如 One-Hot 编码未移除一个类别）。")

if __name__ == "__main__":
    main()
