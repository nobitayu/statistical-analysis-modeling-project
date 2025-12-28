# 4_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import joblib
import os
import config

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def visualize_model():
    """
    生成模型可视化分析图表
    """
    print(">>> [Step 4] 生成模型可视化分析...")
    
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
    
    # 尝试加载模型
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ 已加载模型: {model_path}")
        except:
            pass
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ 已加载标准化器: {scaler_path}")
        except:
            pass
    
    if model is None:
        print("❌ 请先运行模型训练 (3_train_model.py 或 3.5_model_comparison.py)")
        return
    
    # 3. 准备数据
    X = df.drop(columns=['log_views'])
    y = df['log_views']
    
    # 如果模型使用了标准化，需要对X进行标准化
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
        
        # 获取系数
        if hasattr(model, 'params'):
            coefficients = model.params.drop('const', errors='ignore')
        else:
            coefficients = None
    else:
        # sklearn模型
        y_pred = model.predict(X)
        residuals = y - y_pred
        if hasattr(model, 'coef_'):
            coefficients = pd.Series(model.coef_, index=X.columns)
        else:
            coefficients = None
    
    # 5. 创建输出目录
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # 6. 预测值 vs 实际值散点图
    print("正在生成: 预测值 vs 实际值散点图...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, s=20)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    plt.xlabel('实际值 (log_views)', fontsize=12)
    plt.ylabel('预测值 (log_views)', fontsize=12)
    plt.title('预测值 vs 实际值', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    print("✅ 已保存: predicted_vs_actual.png")
    plt.close()
    
    # 7. 残差分析图
    print("正在生成: 残差分析图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 残差 vs 预测值
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('预测值', fontsize=12)
    axes[0].set_ylabel('残差', fontsize=12)
    axes[0].set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q图（正态性检验）
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('残差Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    print("✅ 已保存: residual_analysis.png")
    plt.close()
    
    # 8. 特征重要性（如果有系数）
    if coefficients is not None and len(coefficients) > 0:
        print("正在生成: 特征重要性图...")
        top_n = min(20, len(coefficients))
        coef_abs = coefficients.abs().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        colors = ['red' if coefficients[feat] < 0 else 'blue' 
                 for feat in coef_abs.index]
        plt.barh(range(len(coef_abs)), coef_abs.values, color=colors)
        plt.yticks(range(len(coef_abs)), coef_abs.index)
        plt.xlabel('系数绝对值', fontsize=12)
        plt.title(f'Top {top_n} 特征重要性（按系数绝对值）', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print("✅ 已保存: feature_importance.png")
        plt.close()
    
    # 9. 残差分布直方图
    print("正在生成: 残差分布直方图...")
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('残差', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('残差分布直方图', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差线')
    plt.axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2, label=f'均值: {np.mean(residuals):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'residual_distribution.png'), dpi=300, bbox_inches='tight')
    print("✅ 已保存: residual_distribution.png")
    plt.close()
    
    # 10. 残差统计信息
    print("\n残差统计信息:")
    print(f"  均值: {np.mean(residuals):.6f}")
    print(f"  标准差: {np.std(residuals):.6f}")
    print(f"  最小值: {np.min(residuals):.4f}")
    print(f"  最大值: {np.max(residuals):.4f}")
    
    print(f"\n✅ 所有图表已保存至: {config.PLOTS_DIR}")

if __name__ == "__main__":
    visualize_model()

