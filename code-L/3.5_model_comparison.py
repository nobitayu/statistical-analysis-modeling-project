# 3.5_model_comparison.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import config

def compare_models():
    """
    对比 OLS, Ridge, Lasso, ElasticNet 四种模型
    返回最佳模型和对比结果
    """
    print(">>> [Step 3.5] 多模型对比分析...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {config.PROCESSED_DATA_PATH}")
        return None, None
    
    # 数据清洗
    print("正在清洗数据...")
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.astype(float)
    
    X = df.drop(columns=['log_views'])
    y = df['log_views']
    
    # 2. 标准化（正则化模型需要）
    print("正在标准化特征...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # 3. 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    # 4. 训练多个模型
    models = {}
    results = []
    
    # 4.1 OLS模型
    print("\n[1/4] 训练 OLS 模型...")
    X_train_const = sm.add_constant(X_train, has_constant='add')
    X_test_const = sm.add_constant(X_test, has_constant='add')
    
    try:
        ols_model = sm.OLS(y_train, X_train_const).fit()
        y_pred_ols = ols_model.predict(X_test_const)
        
        models['OLS'] = ols_model
        results.append({
            'Model': 'OLS',
            'R²': r2_score(y_test, y_pred_ols),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ols)),
            'MAE': mean_absolute_error(y_test, y_pred_ols)
        })
        print("✅ OLS 模型训练完成")
    except Exception as e:
        print(f"❌ OLS 模型训练失败: {e}")
    
    # 4.2 Ridge模型（L2正则化）
    print("[2/4] 训练 Ridge 模型（L2正则化）...")
    try:
        # 网格搜索找最优alpha
        ridge_params = {'alpha': [0.1, 1, 10, 100, 1000]}
        ridge_grid = GridSearchCV(
            Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        ridge_model = ridge_grid.best_estimator_
        y_pred_ridge = ridge_model.predict(X_test)
        
        models['Ridge'] = ridge_model
        results.append({
            'Model': f'Ridge (α={ridge_grid.best_params_["alpha"]})',
            'R²': r2_score(y_test, y_pred_ridge),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'MAE': mean_absolute_error(y_test, y_pred_ridge)
        })
        print(f"✅ Ridge 模型训练完成 (最优α={ridge_grid.best_params_['alpha']})")
    except Exception as e:
        print(f"❌ Ridge 模型训练失败: {e}")
    
    # 4.3 Lasso模型（L1正则化，自动特征选择）
    print("[3/4] 训练 Lasso 模型（L1正则化）...")
    try:
        lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        lasso_grid = GridSearchCV(
            Lasso(max_iter=5000), lasso_params, cv=5, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        lasso_grid.fit(X_train, y_train)
        lasso_model = lasso_grid.best_estimator_
        y_pred_lasso = lasso_model.predict(X_test)
        
        # 统计Lasso剔除的特征数
        n_features_kept = np.sum(np.abs(lasso_model.coef_) > 1e-5)
        n_features_removed = len(lasso_model.coef_) - n_features_kept
        
        models['Lasso'] = lasso_model
        results.append({
            'Model': f'Lasso (α={lasso_grid.best_params_["alpha"]}, 保留{n_features_kept}/{len(X.columns)}特征)',
            'R²': r2_score(y_test, y_pred_lasso),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            'MAE': mean_absolute_error(y_test, y_pred_lasso)
        })
        print(f"✅ Lasso 模型训练完成 (最优α={lasso_grid.best_params_['alpha']}, 保留{n_features_kept}特征)")
    except Exception as e:
        print(f"❌ Lasso 模型训练失败: {e}")
    
    # 4.4 ElasticNet模型（L1+L2正则化）
    print("[4/4] 训练 ElasticNet 模型（L1+L2正则化）...")
    try:
        enet_params = {
            'alpha': [0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }
        enet_grid = GridSearchCV(
            ElasticNet(max_iter=5000), enet_params, cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        enet_grid.fit(X_train, y_train)
        enet_model = enet_grid.best_estimator_
        y_pred_enet = enet_model.predict(X_test)
        
        models['ElasticNet'] = enet_model
        results.append({
            'Model': f'ElasticNet (α={enet_grid.best_params_["alpha"]}, l1_ratio={enet_grid.best_params_["l1_ratio"]})',
            'R²': r2_score(y_test, y_pred_enet),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_enet)),
            'MAE': mean_absolute_error(y_test, y_pred_enet)
        })
        print(f"✅ ElasticNet 模型训练完成 (最优α={enet_grid.best_params_['alpha']}, l1_ratio={enet_grid.best_params_['l1_ratio']})")
    except Exception as e:
        print(f"❌ ElasticNet 模型训练失败: {e}")
    
    # 5. 输出对比结果
    if not results:
        print("❌ 所有模型训练失败")
        return None, None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    
    print("\n" + "="*70)
    print("模型对比结果（按R²排序）")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # 6. 保存最佳模型
    best_model_name = results_df.iloc[0]['Model'].split(' ')[0]
    print(f"\n🏆 最佳模型: {best_model_name}")
    
    if best_model_name == 'OLS':
        joblib.dump(models['OLS'], config.MODEL_PATH)
        print(f"✅ OLS模型已保存至: {config.MODEL_PATH}")
    else:
        # 保存sklearn模型和scaler
        best_model_path = config.MODEL_PATH.replace('.pkl', f'_{best_model_name.lower()}.pkl')
        joblib.dump(models[best_model_name], best_model_path)
        scaler_path = best_model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ {best_model_name}模型已保存至: {best_model_path}")
        print(f"✅ 标准化器已保存至: {scaler_path}")
    
    # 保存对比结果
    results_path = config.MODEL_PATH.replace('.pkl', '_comparison.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✅ 对比结果已保存至: {results_path}")
    
    return models, results_df

if __name__ == "__main__":
    compare_models()

