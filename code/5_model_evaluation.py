# 5_model_evaluation.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import joblib
import os
import json
import config

def detailed_evaluation():
    """
    详细的模型评估和统计检验
    """
    print(">>> [Step 5] 详细模型评估...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {config.PROCESSED_DATA_PATH}")
        return
    
    # 数据清洗
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.astype(float)
    
    # 2. 加载模型
    model_path = config.MODEL_PATH
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ 已加载模型: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ 已加载标准化器: {scaler_path}")
        except:
            pass
    
    if model is None:
        print("❌ 请先运行模型训练")
        return
    
    # 3. 准备数据
    X = df.drop(columns=['log_views'])
    y = df['log_views']
    
    # 如果模型使用了标准化
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_scaled
    
    # 4. 获取预测值和残差
    if hasattr(model, 'predict'):
        # statsmodels模型
        X_const = sm.add_constant(X, has_constant='add')
        y_pred = model.predict(X_const)
        residuals = y - y_pred
    else:
        # sklearn模型
        y_pred = model.predict(X)
        residuals = y - y_pred
    
    # 5. 残差正态性检验
    print("\n" + "="*60)
    print("[1] 残差正态性检验")
    print("="*60)
    
    if len(residuals) > 5000:
        # 大样本时使用Kolmogorov-Smirnov检验
        stat, p_value = stats.kstest(
            residuals, 'norm', 
            args=(np.mean(residuals), np.std(residuals))
        )
        test_name = "Kolmogorov-Smirnov"
    else:
        stat, p_value = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
    
    print(f"检验方法: {test_name}")
    print(f"统计量: {stat:.4f}")
    print(f"p值: {p_value:.4f}")
    if p_value > 0.05:
        print("✅ 残差符合正态分布 (p > 0.05)")
        normality_ok = True
    else:
        print("⚠️ 残差不符合正态分布 (p <= 0.05)")
        normality_ok = False
    
    # 6. 异方差检验（仅statsmodels模型）
    heteroscedasticity_ok = None
    if hasattr(model, 'diagn'):
        print("\n" + "="*60)
        print("[2] 异方差检验 (Breusch-Pagan Test)")
        print("="*60)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_const)
            print(f"统计量: {bp_stat:.4f}")
            print(f"p值: {bp_pvalue:.4f}")
            if bp_pvalue > 0.05:
                print("✅ 无异方差问题 (p > 0.05)")
                heteroscedasticity_ok = True
            else:
                print("⚠️ 存在异方差问题 (p <= 0.05)")
                heteroscedasticity_ok = False
        except Exception as e:
            print(f"无法进行异方差检验: {e}")
    
    # 7. 特征重要性分析（仅statsmodels模型）
    print("\n" + "="*60)
    print("[3] 特征重要性分析 (Top 10)")
    print("="*60)
    
    significant_features = 0
    total_features = 0
    
    if hasattr(model, 'params'):
        coef_df = pd.DataFrame({
            '特征': model.params.index,
            '系数': model.params.values,
            'p值': model.pvalues.values
        })
        coef_df = coef_df[coef_df['特征'] != 'const']
        coef_df['系数绝对值'] = coef_df['系数'].abs()
        coef_df = coef_df.sort_values('系数绝对值', ascending=False)
        
        top_10 = coef_df.head(10)
        print(top_10[['特征', '系数', 'p值']].to_string(index=False))
        
        # 统计显著特征
        significant_features = len(coef_df[coef_df['p值'] < 0.05])
        total_features = len(coef_df)
        print(f"\n显著特征数 (p < 0.05): {significant_features} / {total_features}")
    elif hasattr(model, 'coef_'):
        # sklearn模型
        coef_df = pd.DataFrame({
            '特征': X.columns,
            '系数': model.coef_,
            '系数绝对值': np.abs(model.coef_)
        })
        coef_df = coef_df.sort_values('系数绝对值', ascending=False)
        top_10 = coef_df.head(10)
        print(top_10[['特征', '系数']].to_string(index=False))
        total_features = len(coef_df)
    
    # 8. 模型性能总结
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
    
    print("\n" + "="*60)
    print("[4] 模型性能总结")
    print("="*60)
    print(f"R² (决定系数):           {r2:.4f}")
    print(f"RMSE (均方根误差):        {rmse:.4f}")
    print(f"MAE (平均绝对误差):       {mae:.4f}")
    print(f"MAPE (平均百分比误差):     {mape:.2f}%")
    print(f"样本数:                  {len(y)}")
    print(f"特征数:                  {total_features}")
    
    # 9. 保存评估报告
    report = {
        'model_performance': {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        },
        'residual_analysis': {
            'normality_test': test_name,
            'normality_statistic': float(stat),
            'normality_p_value': float(p_value),
            'normality_ok': normality_ok,
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals))
        },
        'model_info': {
            'n_samples': int(len(y)),
            'n_features': int(total_features),
            'n_significant_features': int(significant_features) if significant_features > 0 else None
        }
    }
    
    if heteroscedasticity_ok is not None:
        report['residual_analysis']['heteroscedasticity_ok'] = heteroscedasticity_ok
    
    os.makedirs(os.path.dirname(config.EVALUATION_REPORT_PATH), exist_ok=True)
    with open(config.EVALUATION_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 评估报告已保存至: {config.EVALUATION_REPORT_PATH}")
    
    return report

if __name__ == "__main__":
    detailed_evaluation()

