# 2_check_vif.py (强力清洗修复版)
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import config

def check_vif():
    print(">>> [Step 2] 开始共线性(VIF)检查...")
    
    # 1. 读取清洗后的数据
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {config.PROCESSED_DATA_PATH}")
        print("请先运行 1_preprocess.py")
        return
    
    # 剔除 Y，只看 X
    # 确保 log_views 在列中，防止报错
    if 'log_views' in df.columns:
        X = df.drop(columns=['log_views'])
    else:
        X = df.copy()

    # ================= 修复核心：强制类型转换 =================
    print(f"原始特征数量: {X.shape[1]}")
    
    # 1. 检查是否有 Object (字符串) 类型的列
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"⚠️ 警告：检测到非数值类型的列: {obj_cols}")
        print("尝试自动转换为数值...")
        
    # 2. 强制将所有数据转换为 float (数字)
    # coerce 参数会将无法转换的字符串变成 NaN (空值)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 3. 检查是否有转换失败的列 (变成了全是 NaN)
    if X.isnull().any().any():
        print("⚠️ 检测到数据中包含空值或无法转换的字符，正在填充为 0...")
        X.fillna(0, inplace=True)

    # 4. 再次确保数据类型为 float
    X = X.astype(float)
    
    # 5. 剔除方差为 0 的列 (全是一样的值，计算 VIF 会报错)
    # 比如某列全是 0，会导致除以零错误
    var_check = X.var()
    drop_cols = var_check[var_check == 0].index.tolist()
    if drop_cols:
        print(f"⚠️ 剔除方差为0的常量列 (对模型无用): {drop_cols}")
        X = X.drop(columns=drop_cols)

    # ========================================================
    
    # 必须手动添加常数项来计算 VIF
    # 如果数据中有无穷大 inf，这一步会报错，所以先处理 inf
    X.replace([np.inf, -np.inf], 0, inplace=True)
    
    try:
        X_const = add_constant(X)
    except Exception as e:
        print(f"❌ 添加常数项失败，原因: {e}")
        return
    
    # 2. 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    
    print(f"正在计算 VIF (特征数: {len(X_const.columns)})...")
    
    # 列表推导式计算 VIF
    vif_list = []
    for i in range(len(X_const.columns)):
        try:
            val = variance_inflation_factor(X_const.values, i)
            vif_list.append(val)
        except Exception as e:
            # 如果计算出错（通常是因为完美共线性导致除以0），设为无穷大
            vif_list.append(float('inf'))
            
    vif_data["VIF"] = vif_list
    
    # 3. 排序并显示
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\n" + "="*40)
    print("VIF 结果 (VIF > 10 表示严重共线性)")
    print("="*40)
    print(vif_data)
    print("="*40)
    
    # 自动建议
    # 忽略 const 列
    high_vif = vif_data[(vif_data['VIF'] > 10) & (vif_data['Feature'] != 'const')]
    
    if not high_vif.empty:
        print("⚠️ 警告: 以下变量存在严重共线性，建议在 config.py 中移除其中一个:")
        print(high_vif['Feature'].tolist())
    else:
        print("✅ 完美！所有特征 VIF 均在安全范围内，可以直接建模。")

if __name__ == "__main__":
    check_vif()