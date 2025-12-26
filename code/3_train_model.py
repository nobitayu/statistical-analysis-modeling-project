# 3_train_model.py (增强版：加入交叉验证、更多评估指标、特征标准化)
import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import config

def train_ols(use_scaling=None, use_cv=None):
    """
    训练OLS回归模型
    
    参数:
        use_scaling: 是否使用特征标准化（None时使用config配置）
        use_cv: 是否使用交叉验证（None时使用config配置）
    """
    print(">>> [Step 3] 开始训练 OLS 回归模型...")
    
    # 使用配置文件的默认值
    if use_scaling is None:
        use_scaling = config.USE_FEATURE_SCALING
    if use_cv is None:
        use_cv = config.USE_CROSS_VALIDATION
    
    # 1. 读取数据
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {config.PROCESSED_DATA_PATH}")
        return None
    
    # ================= 数据清洗 =================
    print("正在清洗数据以确保全是数字...")
    
    # 强制转换为数值类型 (无法转换的变 NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 填充空值 (NaN) 为 0
    df.fillna(0, inplace=True)
    
    # 确保没有无穷大 (inf)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # 再次确保数据类型为 float
    df = df.astype(float)
    # ========================================================

    # 2. 准备 X 和 y
    if 'log_views' not in df.columns:
        print("❌ 错误：数据中找不到目标列 'log_views'。请重新运行 Step 1。")
        return None

    X = df.drop(columns=['log_views'])
    y = df['log_views']
    
    # 3. 特征标准化（可选）
    scaler = None
    if use_scaling:
        print("正在标准化特征...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_scaled
        print("✅ 特征标准化完成")
    
    # 4. 切分训练集和测试集 (80% 训练, 20% 验证)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE
        )
    except ValueError as e:
        print(f"❌ 数据切分失败: {e}")
        return None
    
    # 5. 添加常数项 (Intercept) - 统计模型必须手动加
    X_train_const = sm.add_constant(X_train, has_constant='add')
    X_test_const = sm.add_constant(X_test, has_constant='add')
    
    print(f"训练集形状: {X_train_const.shape}")
    print(f"测试集形状: {X_test_const.shape}")
    
    # 6. 训练模型
    try:
        model = sm.OLS(y_train, X_train_const).fit()
    except Exception as e:
        print(f"❌ 模型训练失败，原因可能是数据中仍有脏数据: {e}")
        return None
    
    # 7. 交叉验证（可选）
    if use_cv:
        print(f"\n>>> 进行{config.CV_FOLDS}折交叉验证...")
        kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        cv_r2_scores = []
        cv_rmse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            X_cv_train = X_train_const.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train_const.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            cv_model = sm.OLS(y_cv_train, X_cv_train).fit()
            y_cv_pred = cv_model.predict(X_cv_val)
            
            cv_r2 = r2_score(y_cv_val, y_cv_pred)
            cv_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
            
            cv_r2_scores.append(cv_r2)
            cv_rmse_scores.append(cv_rmse)
        
        print(f"交叉验证 R² 得分: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")
        print(f"交叉验证 RMSE:    {np.mean(cv_rmse_scores):.4f} (±{np.std(cv_rmse_scores):.4f})")
    
    # 8. 输出详细报表
    print("\n" + "#"*50)
    print("OLS Regression Results (请截图保存下方表格)")
    print("#"*50)
    print(model.summary())
    
    # 9. 测试集评估（增强指标）
    try:
        y_pred = model.predict(X_test_const)
        
        # 计算多个评估指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        print("\n" + "="*50)
        print("[测试集评估结果]")
        print("="*50)
        print(f"R² (决定系数):           {r2:.4f}")
        print(f"RMSE (均方根误差):        {rmse:.4f}")
        print(f"MAE (平均绝对误差):       {mae:.4f}")
        print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"评估失败: {e}")
    
    # 10. 保存模型和标准化器
    joblib.dump(model, config.MODEL_PATH)
    print(f"\n✅ 模型已保存至: {config.MODEL_PATH}")
    
    if scaler:
        scaler_path = config.MODEL_PATH.replace('.pkl', '_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ 标准化器已保存至: {scaler_path}")
    
    print("你可以把这个 .pkl 文件交给后端同学用于预测系统。")
    
    return model, scaler

if __name__ == "__main__":
    train_ols()