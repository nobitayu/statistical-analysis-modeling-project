"""
两阶段栏栅模型（Hurdle Model）第二阶段：回归模型（The Quantifier）
=====================================================================

本脚本实现严谨的多元线性回归分析，用于预测潜在热门视频的播放量。

作者：统计分析团队
日期：2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Statsmodels 相关导入
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置图表风格（使用英文标签以避免中文显示问题）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# Step 0: 数据加载与预处理
# ============================================================================

def load_and_prepare_data(filepath):
    """
    加载数据并进行基础预处理
    
    Parameters:
    -----------
    filepath : str
        数据文件路径
    
    Returns:
    --------
    df : DataFrame
        预处理后的数据框
    """
    print("=" * 80)
    print("Step 0: Data Loading and Preprocessing")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")
    
    # 检查目标变量
    if 'view_count' in df.columns:
        df['views'] = df['view_count']
    elif 'views' not in df.columns:
        raise ValueError("Data must contain 'view_count' or 'views' column")
    
    # 筛选非零播放量（符合截断回归逻辑）
    initial_count = len(df)
    df = df[df['views'] > 0].copy()
    print(f"Filtered data shape: {df.shape} (removed {initial_count - len(df)} zero values)")
    
    return df


# ============================================================================
# Step 1: 变量变换与特征工程
# ============================================================================

def feature_engineering(df):
    # 执行变量变换和特征工程
    
    # 统计学意义：
    # - Log变换：降低偏度，使数据更接近正态分布，满足线性回归的正态性假设
    # - 使用原始互动数（对数变换后）而非互动率，避免反向因果关系
    #   因为 like_rate = likes / view_count，comment_rate = comment_count / view_count
    #   互动率的分母就是目标变量，会导致负相关（反向因果关系）
    
    print("\n" + "=" * 80)
    print("Step 1: Variable Transformation and Feature Engineering")
    print("=" * 80)
    
    # 1.1 目标变量 Log 变换
    # 使用 log1p = log(1+x) 以避免 log(0) 的问题
    df['log_views'] = np.log1p(df['views'])
    print(f"\n✓ Target variable 'views' has been log-transformed (log1p)")
    print(f"  Original views stats: mean={df['views'].mean():.2f}, median={df['views'].median():.2f}")
    print(f"  Log-transformed stats: mean={df['log_views'].mean():.4f}, median={df['log_views'].median():.4f}")
    
    # 注意：不使用点赞和评论相关特征，因为这些是视频发布后的数据，不能用于预测
    # 只使用视频发布前可以获得的元数据特征
    
    return df


# ============================================================================
# Step 2: 建立 OLS 模型 (Base Model)
# ============================================================================

def build_ols_model(df):
    """
    构建多元线性回归模型
    
    统计学意义：
    - OLS (Ordinary Least Squares) 是最基础的线性回归方法
    - R²: 模型解释的方差比例
    - F-statistic: 整体模型显著性检验
    - P-values: 各系数的显著性检验
    
    注意：对于分类特征组（category_* 和 period_*），根据文档：
    - category_24 已删除作为参照组
    - period_Afternoon 已删除作为参照组
    这是标准的虚拟变量处理方式，避免完全多重共线性
    
    Parameters:
    -----------
    df : DataFrame
        特征工程后的数据框
    
    Returns:
    --------
    model : RegressionResults
        拟合的模型对象
    formula : str
        使用的回归公式
    """
    print("\n" + "=" * 80)
    print("Step 2: Building OLS Model (Base Model)")
    print("=" * 80)
    
    # 选择特征变量
    # 根据文档，我们使用主要的特征类别
    feature_vars = []
    
    # 分类特征 (category_*)
    # 注意：category_24 已作为参照组删除，不需要处理
    category_cols = [col for col in df.columns if col.startswith('category_')]
    feature_vars.extend(category_cols)
    print(f"\n✓ Added {len(category_cols)} category features (category_24 is reference group, excluded)")
    
    # 时间特征
    # 注意：period_Afternoon 已作为参照组删除，不需要处理
    time_cols = ['period_Dawn', 'period_Evening', 'period_Morning', 'is_weekend']
    available_time_cols = [col for col in time_cols if col in df.columns]
    feature_vars.extend(available_time_cols)
    print(f"✓ Added {len(available_time_cols)} time features (period_Afternoon is reference group, excluded)")
    
    # 互动与标题特征
    # 注意：不使用 log_likes 和 log_comment_count，因为这些是视频发布后的数据，不能用于预测
    # 只使用视频发布前可以获得的特征
    interaction_cols = ['title_length', 'title_upper_ratio', 'title_has_punct']
    available_interaction_cols = [col for col in interaction_cols if col in df.columns]
    feature_vars.extend(available_interaction_cols)
    print(f"✓ Added {len(available_interaction_cols)} title features")
    print("  (Note: Excluding likes/comments as they are post-publication data)")
    
    # 频道特征（使用对数变换版本）
    # 注意：log_channel_avg_comment_count 可能也包含发布后数据，但频道历史平均数据可以作为预测特征
    channel_cols = ['log_channel_activity', 'log_channel_avg_views', 
                   'log_channel_avg_comment_count', 'channel_name_len']
    available_channel_cols = [col for col in channel_cols if col in df.columns]
    feature_vars.extend(available_channel_cols)
    print(f"✓ Added {len(available_channel_cols)} channel features")
    
    # 文本衍生特征
    text_cols = ['log_tags_count', 'tag_density', 'log_desc_length', 
                'desc_has_timestamp', 'desc_keyword_count']
    available_text_cols = [col for col in text_cols if col in df.columns]
    feature_vars.extend(available_text_cols)
    print(f"✓ Added {len(available_text_cols)} text-derived features")
    
    # 注意：like_rate 和 comment_rate 已经在 interaction_cols 中，不需要额外添加
    
    # 移除不在数据中的变量
    feature_vars = [var for var in feature_vars if var in df.columns]
    
    # 检查缺失值
    missing = df[feature_vars + ['log_views']].isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Found missing values, will remove rows with missing values:")
        print(missing[missing > 0])
        df = df.dropna(subset=feature_vars + ['log_views'])
        print(f"  Data shape after removal: {df.shape}")
    
    # 构建回归公式
    # 格式: "log_views ~ var1 + var2 + ..."
    formula = "log_views ~ " + " + ".join(feature_vars)
    print(f"\nRegression formula contains {len(feature_vars)} feature variables")
    print(f"Formula preview: log_views ~ ... + log_likes + log_comment_count")
    
    # 拟合 OLS 模型
    print("\nFitting OLS model...")
    model = ols(formula, data=df).fit()
    
    # 输出详细统计摘要
    print("\n" + "=" * 80)
    print("OLS Model Statistical Summary")
    print("=" * 80)
    print(model.summary())
    
    # 提取关键统计量
    print("\n" + "-" * 80)
    print("Key Statistics Summary:")
    print("-" * 80)
    print(f"R-squared (R²):                    {model.rsquared:.4f}")
    print(f"Adjusted R-squared:                 {model.rsquared_adj:.4f}")
    print(f"F-statistic:                       {model.fvalue:.4f}")
    print(f"F-statistic p-value:               {model.f_pvalue:.2e}")
    print(f"Model significance:                 {'Significant' if model.f_pvalue < 0.05 else 'Not significant'} (α=0.05)")
    
    # 显示关键显著变量的系数
    print(f"\nKey Significant Variables (p < 0.05):")
    significant_vars = model.params[model.pvalues < 0.05].index.tolist()
    significant_vars = [v for v in significant_vars if v != 'Intercept']
    for var in significant_vars[:10]:  # 显示前10个
        coef = model.params[var]
        pval = model.pvalues[var]
        print(f"  {var}: {coef:.6f} (p={pval:.4f})")
    
    # 保存模型系数到文件（用于原型系统）
    save_model_coefficients(model, formula)
    
    return model, formula, df


# ============================================================================
# 保存模型系数（用于原型系统）
# ============================================================================

def save_model_coefficients(model, formula):
    """
    保存模型系数到文件，供原型系统使用
    
    Parameters:
    -----------
    model : RegressionResults
        拟合的模型（可能是稳健标准误模型）
    formula : str
        回归公式
    """
    import json
    
    # 安全地获取参数名称和值
    # 对于稳健标准误模型，params 可能是数组而不是 Series
    try:
        # 尝试获取参数名称
        if hasattr(model.params, 'index'):
            param_names = list(model.params.index)
            param_values = model.params.values if hasattr(model.params, 'values') else np.array(model.params)
            bse_values = model.bse.values if hasattr(model.bse, 'values') else np.array(model.bse)
            pval_values = model.pvalues.values if hasattr(model.pvalues, 'values') else np.array(model.pvalues)
        else:
            # 如果是数组，从原始模型获取名称
            param_names = list(model.model.exog_names)
            param_values = np.array(model.params)
            bse_values = np.array(model.bse)
            pval_values = np.array(model.pvalues)
    except:
        # 最后的备选方案：从原始模型获取
        param_names = list(model.model.exog_names)
        param_values = np.array(model.params)
        bse_values = np.array(model.bse)
        pval_values = np.array(model.pvalues)
    
    # 找到截距的索引
    intercept_idx = None
    for i, name in enumerate(param_names):
        if name == 'Intercept' or name == 'const':
            intercept_idx = i
            break
    
    if intercept_idx is None:
        intercept_idx = 0  # 默认第一个是截距
    
    # 创建系数字典
    coefficients_dict = {
        'intercept': float(param_values[intercept_idx]),
        'coefficients': {},
        'formula': formula,
        'model_info': {
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'n_observations': int(model.nobs),
            'n_features': len(param_names) - 1
        }
    }
    
    # 保存所有系数（排除截距）
    for i, var in enumerate(param_names):
        if i != intercept_idx and var not in ['Intercept', 'const']:
            coefficients_dict['coefficients'][var] = {
                'coefficient': float(param_values[i]),
                'std_err': float(bse_values[i]),
                'p_value': float(pval_values[i]),
                'is_significant': bool(pval_values[i] < 0.05)
            }
    
    # 保存为 JSON 文件
    with open('model_coefficients.json', 'w', encoding='utf-8') as f:
        json.dump(coefficients_dict, f, indent=2, ensure_ascii=False)
    
    # 同时保存为 CSV 文件（便于查看）
    coeff_df = pd.DataFrame({
        'variable': param_names,
        'coefficient': param_values,
        'std_err': bse_values,
        'p_value': pval_values,
        'is_significant': pval_values < 0.05
    })
    coeff_df.to_csv('model_coefficients.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Model coefficients saved to:")
    print(f"  - model_coefficients.json (for prototype system)")
    print(f"  - model_coefficients.csv (for reference)")
    
    return coefficients_dict


# ============================================================================
# Step 3: 模型诊断 (Diagnostics)
# ============================================================================

def model_diagnostics(model, df):
    """
    执行全面的模型诊断
    
    包括：
    1. 残差分析（残差 vs 拟合值图、Q-Q图）
    2. 异方差检验（Breusch-Pagan Test）
    3. 如果存在异方差，使用稳健标准误重新拟合
    
    Parameters:
    -----------
    model : RegressionResults
        初始 OLS 模型
    df : DataFrame
        数据框
    
    Returns:
    --------
    robust_model : RegressionResults or None
        如果存在异方差，返回使用稳健标准误的模型；否则返回 None
    """
    print("\n" + "=" * 80)
    print("Step 3: Model Diagnostics")
    print("=" * 80)
    
    # 获取残差和拟合值
    fitted_values = model.fittedvalues
    residuals = model.resid
    standardized_residuals = residuals / np.sqrt(model.mse_resid)
    
    # 3.1 残差分析可视化
    print("\n3.1 Residual Analysis Visualization")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Diagnostics', fontsize=16, fontweight='bold')
    
    # 3.1.1 残差 vs 拟合值图
    ax1 = axes[0, 0]
    ax1.scatter(fitted_values, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual line')
    ax1.set_xlabel('Fitted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加平滑曲线（使用移动平均）
    try:
        # 对数据进行排序
        sorted_idx = np.argsort(fitted_values)
        sorted_fitted = fitted_values[sorted_idx]
        sorted_residuals = residuals[sorted_idx]
        
        # 使用移动平均创建平滑曲线
        window_size = max(50, len(fitted_values) // 20)  # 窗口大小为数据点的5%
        if window_size % 2 == 0:
            window_size += 1  # 确保是奇数
        
        from scipy.signal import savgol_filter
        if len(sorted_residuals) > window_size:
            smooth_residuals = savgol_filter(sorted_residuals, window_size, 3)
            ax1.plot(sorted_fitted, smooth_residuals, 
                    'g-', linewidth=2, label='Smooth curve', alpha=0.7)
            ax1.legend()
    except Exception as e:
        # 如果平滑失败，尝试简单的移动平均
        try:
            sorted_idx = np.argsort(fitted_values)
            sorted_fitted = fitted_values[sorted_idx]
            sorted_residuals = residuals[sorted_idx]
            
            # 简单的移动平均
            window = max(10, len(fitted_values) // 50)
            if len(sorted_residuals) > window * 2:
                smooth = np.convolve(sorted_residuals, np.ones(window)/window, mode='valid')
                smooth_fitted = sorted_fitted[window//2:len(sorted_fitted)-window//2+1]
                if len(smooth) == len(smooth_fitted):
                    ax1.plot(smooth_fitted, smooth, 
                            'g-', linewidth=2, label='Smooth curve', alpha=0.7)
                    ax1.legend()
        except:
            pass  # 如果都失败，就不显示平滑曲线
    
    # 3.1.2 Q-Q 图（正态性检验）
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3.1.3 标准化残差 vs 拟合值
    ax3 = axes[1, 0]
    ax3.scatter(fitted_values, standardized_residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='±2σ')
    ax3.axhline(y=-2, color='orange', linestyle='--', linewidth=1)
    ax3.set_xlabel('Fitted Values', fontsize=12)
    ax3.set_ylabel('Standardized Residuals', fontsize=12)
    ax3.set_title('Standardized Residuals vs Fitted', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 3.1.4 残差直方图
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
    # 叠加正态分布曲线
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
            label=f'Normal dist (μ={mu:.3f}, σ={sigma:.3f})')
    ax4.set_xlabel('Residuals', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Residual Distribution Histogram', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Diagnostic plots saved as 'model_diagnostics.png'")
    plt.show()
    
    # 3.1.5 额外的可视化：系数重要性图和实际值 vs 预测值
    visualize_coefficient_importance(model)
    visualize_actual_vs_predicted(model, df)
    
    # 3.2 正态性检验（Shapiro-Wilk Test，适用于小样本）
    print("\n3.2 Residual Normality Test")
    print("-" * 80)
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk Test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_p:.4f}")
        print(f"  Conclusion: {'Residuals approximately normal' if shapiro_p > 0.05 else 'Residuals significantly deviate from normal'} (α=0.05)")
    else:
        print("Sample size too large, skipping Shapiro-Wilk test (only for n≤5000)")
        # 使用 Kolmogorov-Smirnov 检验
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(mu, sigma))
        print(f"Kolmogorov-Smirnov Test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.4f}")
        print(f"  Conclusion: {'Residuals approximately normal' if ks_p > 0.05 else 'Residuals significantly deviate from normal'} (α=0.05)")
    
    # 3.3 异方差检验（Breusch-Pagan Test）
    print("\n3.3 Heteroscedasticity Test (Breusch-Pagan Test)")
    print("-" * 80)
    
    # 执行 Breusch-Pagan 检验
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(
        residuals, model.model.exog
    )
    
    print(f"Breusch-Pagan LM statistic: {bp_lm:.4f}")
    print(f"Breusch-Pagan LM p-value: {bp_lm_pvalue:.4f}")
    print(f"Breusch-Pagan F statistic: {bp_fvalue:.4f}")
    print(f"Breusch-Pagan F p-value: {bp_f_pvalue:.4f}")
    
    if bp_lm_pvalue < 0.05:
        print(f"\n⚠ Heteroscedasticity detected (p-value = {bp_lm_pvalue:.4f} < 0.05)")
        print("  Will refit model with robust standard errors (HC3)...")
        
        # 使用稳健标准误重新拟合
        robust_model = model.get_robustcov_results(cov_type='HC3')
        
        print("\n" + "=" * 80)
        print("Model Summary with Robust Standard Errors (HC3)")
        print("=" * 80)
        print(robust_model.summary())
        
        print("\n" + "-" * 80)
        print("Key Statistics Comparison:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Original Model':<20} {'Robust SE Model':<20}")
        print("-" * 70)
        print(f"{'R-squared':<30} {model.rsquared:<20.4f} {robust_model.rsquared:<20.4f}")
        print(f"{'F-statistic':<30} {model.fvalue:<20.4f} {robust_model.fvalue:<20.4f}")
        
        # 比较关键显著变量的标准误（显示前5个最显著的）
        significant_vars = model.params[model.pvalues < 0.05].index.tolist()
        significant_vars = [v for v in significant_vars if v != 'Intercept']
        significant_vars = sorted(significant_vars, key=lambda x: model.pvalues[x])[:5]  # 前5个最显著的
        
        if significant_vars:
            print(f"\nTop 5 Significant Variables Coefficient Comparison:")
            for var in significant_vars:
                print(f"\n{var}:")
                print(f"  Original model coefficient: {model.params[var]:.6f}")
                print(f"  Original model SE: {model.bse[var]:.6f}")
                try:
                    robust_se = robust_model.bse.loc[var] if hasattr(robust_model.bse, 'loc') else robust_model.bse[var]
                    robust_pval = robust_model.pvalues.loc[var] if hasattr(robust_model.pvalues, 'loc') else robust_model.pvalues[var]
                except:
                    param_names = list(model.params.index)
                    if var in param_names:
                        idx = param_names.index(var)
                        robust_se = robust_model.bse[idx]
                        robust_pval = robust_model.pvalues[idx]
                    else:
                        robust_se = None
                        robust_pval = None
                
                if robust_se is not None:
                    print(f"  Robust SE: {robust_se:.6f}")
                    print(f"  Original model p-value: {model.pvalues[var]:.4f}")
                    print(f"  Robust model p-value: {robust_pval:.4f}")
        
        return robust_model
    else:
        print(f"\n✓ No significant heteroscedasticity detected (p-value = {bp_lm_pvalue:.4f} ≥ 0.05)")
        print("  Model satisfies homoscedasticity assumption, no need for robust standard errors")
        return None


# ============================================================================
# 系数重要性可视化
# ============================================================================

def visualize_coefficient_importance(model):
    """
    可视化模型系数的重要性
    
    Parameters:
    -----------
    model : RegressionResults
        拟合的模型
    """
    print("\n3.1.5 Coefficient Importance Visualization")
    print("-" * 80)
    
    # 获取系数（排除截距）
    coef_data = []
    for var in model.params.index:
        if var != 'Intercept':
            coef_data.append({
                'variable': var,
                'coefficient': model.params[var],
                'abs_coefficient': abs(model.params[var]),
                'p_value': model.pvalues[var],
                'is_significant': model.pvalues[var] < 0.05
            })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 上图：系数大小（按绝对值排序，显示前20个）
    ax1 = axes[0]
    top_coefs = coef_df.head(20)
    colors = ['red' if not sig else 'steelblue' for sig in top_coefs['is_significant']]
    bars = ax1.barh(range(len(top_coefs)), top_coefs['coefficient'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_coefs)))
    ax1.set_yticklabels(top_coefs['variable'], fontsize=9)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Coefficient Value', fontsize=12)
    ax1.set_title('Top 20 Variable Coefficients (Sorted by Absolute Value)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend([plt.Rectangle((0,0),1,1, facecolor='steelblue', alpha=0.7), 
                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7)],
               ['Significant (p<0.05)', 'Not Significant'], loc='best')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(top_coefs.iterrows()):
        ax1.text(row['coefficient'], i, f"  {row['coefficient']:.4f}", 
                va='center', fontsize=8)
    
    # 下图：系数 vs P值散点图
    ax2 = axes[1]
    scatter = ax2.scatter(coef_df['coefficient'], -np.log10(coef_df['p_value'] + 1e-10),
                         c=coef_df['is_significant'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Coefficient Value', fontsize=12)
    ax2.set_ylabel('-log10(p-value)', fontsize=12)
    ax2.set_title('Coefficient vs Significance (Volcano Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # 标注最显著的变量
    top_sig = coef_df.nsmallest(5, 'p_value')
    for _, row in top_sig.iterrows():
        ax2.annotate(row['variable'], 
                    (row['coefficient'], -np.log10(row['p_value'] + 1e-10)),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('coefficient_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Coefficient importance plots saved as 'coefficient_importance.png'")
    plt.show()


# ============================================================================
# 实际值 vs 预测值可视化
# ============================================================================

def visualize_actual_vs_predicted(model, df):
    """
    可视化实际值 vs 预测值
    
    Parameters:
    -----------
    model : RegressionResults
        拟合的模型
    df : DataFrame
        数据框
    """
    print("\n3.1.6 Actual vs Predicted Values Visualization")
    print("-" * 80)
    
    # 获取实际值和预测值
    actual_log = df['log_views'].values
    predicted_log = model.fittedvalues.values
    
    # 转换回原始尺度
    actual_views = np.expm1(actual_log)
    predicted_views = np.expm1(predicted_log)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：对数尺度
    ax1 = axes[0]
    ax1.scatter(actual_log, predicted_log, alpha=0.5, s=10, edgecolors='k', linewidth=0.3)
    
    # 添加完美预测线（y=x）
    min_val = min(actual_log.min(), predicted_log.min())
    max_val = max(actual_log.max(), predicted_log.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # 计算 R²
    r2 = r2_score(actual_log, predicted_log)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Actual log(views)', fontsize=12)
    ax1.set_ylabel('Predicted log(views)', fontsize=12)
    ax1.set_title('Actual vs Predicted (Log Scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 右图：原始尺度
    ax2 = axes[1]
    # 为了可视化效果，限制范围（避免极端值）
    max_display = np.percentile(actual_views, 95)
    mask = (actual_views <= max_display) & (predicted_views <= max_display)
    
    ax2.scatter(actual_views[mask], predicted_views[mask], alpha=0.5, s=10, edgecolors='k', linewidth=0.3)
    
    # 添加完美预测线
    min_val = min(actual_views[mask].min(), predicted_views[mask].min())
    max_val = max(actual_views[mask].max(), predicted_views[mask].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # 计算 R²（原始尺度）
    r2_orig = r2_score(actual_views, predicted_views)
    ax2.text(0.05, 0.95, f'R² = {r2_orig:.4f}', transform=ax2.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.05, 0.88, f'Displaying 95% of data', transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', style='italic')
    
    ax2.set_xlabel('Actual Views', fontsize=12)
    ax2.set_ylabel('Predicted Views', fontsize=12)
    ax2.set_title('Actual vs Predicted (Original Scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✓ Actual vs predicted plots saved as 'actual_vs_predicted.png'")
    plt.show()


# ============================================================================
# Step 4: 预测与区间估计 (Inference)
# ============================================================================

def prediction_with_intervals(model, df, scenario='typical'):
    """
    对新样本进行预测，并计算预测区间
    
    统计学意义：
    - 点预测：基于模型系数的期望值预测
    - 预测区间：考虑模型不确定性和随机误差的区间估计
    - 95% 预测区间意味着有 95% 的概率真实值落在此区间内
    
    Parameters:
    -----------
    model : RegressionResults
        拟合的模型（可能是稳健标准误模型）
    df : DataFrame
        训练数据框
    scenario : str
        预测场景：'typical' (典型), 'high_potential' (高潜力), 'low_potential' (低潜力)
    
    Returns:
    --------
    prediction_dict : dict
        包含预测结果的字典
    """
    print("\n" + "=" * 80)
    print("Step 4: Prediction and Interval Estimation (Inference)")
    print("=" * 80)
    
    # 创建新样本的特征值
    print(f"\nCreating simulated new sample (Scenario: {scenario}):")
    
    # 根据场景设置特征值
    if scenario == 'high_potential':
        # 高潜力场景：使用上四分位数（对于正向特征）或下四分位数（对于负向特征）
        print("  Using high potential values (75th percentile for positive features)")
    elif scenario == 'low_potential':
        # 低潜力场景：使用下四分位数（对于正向特征）或上四分位数（对于负向特征）
        print("  Using low potential values (25th percentile for positive features)")
    else:
        # 典型场景：使用中位数
        print("  Using typical values (median)")
    
    # 创建新样本的 DataFrame
    # 注意：使用 formula API 时，必须传递包含所有变量的 DataFrame
    # 最可靠的方法：从原始训练数据复制一行，然后修改需要的值
    # 这样可以确保所有变量都存在且格式正确
    
    # 从原始数据中取第一行作为模板（排除目标变量）
    # 然后修改我们需要的值
    new_sample = df.iloc[[0]].copy()  # 使用 [[0]] 保持 DataFrame 格式
    
    # 移除目标变量列（如果存在）
    if 'log_views' in new_sample.columns:
        new_sample = new_sample.drop(columns=['log_views'])
    if 'views' in new_sample.columns:
        new_sample = new_sample.drop(columns=['views'])
    if 'view_count' in new_sample.columns:
        new_sample = new_sample.drop(columns=['view_count'])
    
    # 安全地获取模型参数名称列表
    # 处理稳健标准误模型（params 可能是数组）
    try:
        if hasattr(model.params, 'index'):
            model_var_names = list(model.params.index)
            # 创建参数名到系数的映射
            param_dict = {name: model.params[name] for name in model_var_names}
        else:
            # 如果是数组，从原始模型获取名称
            model_var_names = list(model.model.exog_names)
            param_values = np.array(model.params)
            param_dict = {name: param_values[i] for i, name in enumerate(model_var_names)}
    except:
        # 最后的备选方案
        model_var_names = list(model.model.exog_names)
        param_values = np.array(model.params)
        param_dict = {name: param_values[i] for i, name in enumerate(model_var_names)}
    
    # 设置所有分类变量为0（参照组）
    # 同时处理字符串列和数值列
    for col in new_sample.columns:
        if col.startswith('category_'):
            new_sample[col] = 0
        elif col.startswith('period_'):
            new_sample[col] = 0
        elif col == 'is_weekend':
            new_sample[col] = 0
        elif col == 'title_has_punct':
            new_sample[col] = 0
        elif col == 'desc_has_timestamp':
            new_sample[col] = 0
        elif col == 'desc_has_youtube_link':
            new_sample[col] = 0
        elif col == 'channel_has_digit':
            new_sample[col] = 0
        elif col == 'channel_has_special':
            new_sample[col] = 0
        elif col in ['title', 'publishedAt', 'trending_date', 'tags', 'categoryId']:
            # 字符串列或ID列：保持原值（这些列不在模型中，但需要存在以避免错误）
            # 实际上，如果这些列不在公式中，patsy 会忽略它们
            pass  # 保持原值
        else:
            # 其他连续变量：根据场景设置值
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # 检查变量是否在模型中
                if col in param_dict:
                    coef = param_dict[col]
                    # 根据场景和系数方向设置值
                    if scenario == 'high_potential':
                        # 高潜力：正向系数用上四分位数，负向系数用下四分位数
                        if coef > 0:
                            new_sample[col] = df[col].quantile(0.75)
                        else:
                            new_sample[col] = df[col].quantile(0.25)
                    elif scenario == 'low_potential':
                        # 低潜力：正向系数用下四分位数，负向系数用上四分位数
                        if coef > 0:
                            new_sample[col] = df[col].quantile(0.25)
                        else:
                            new_sample[col] = df[col].quantile(0.75)
                    else:
                        # 典型场景：使用中位数
                        new_sample[col] = df[col].median()
                else:
                    # 变量不在模型中，使用中位数
                    new_sample[col] = df[col].median()
                # 如果是非数值列（字符串、日期等），保持原值
    
    # 使用 DataFrame 进行预测（formula API 要求）
    # formula API 会自动从 DataFrame 中提取公式需要的变量
    X_new = new_sample
    
    # 点预测（对数尺度）
    log_pred = model.predict(X_new)[0]
    
    # 预测区间（对数尺度）
    # 使用 get_prediction 方法获取预测区间
    pred_result = model.get_prediction(X_new)
    pred_ci = pred_result.conf_int(alpha=0.05)  # 95% 置信区间
    
    # 注意：这里得到的是置信区间，不是预测区间
    # 预测区间需要考虑残差的标准误
    # 计算预测区间的标准误
    mse = model.mse_resid
    pred_se = np.sqrt(mse + pred_result.var_pred_mean[0])
    
    # 95% 预测区间（对数尺度）
    t_critical = stats.t.ppf(0.975, model.df_resid)
    log_pred_lower = log_pred - t_critical * pred_se
    log_pred_upper = log_pred + t_critical * pred_se
    
    # 转换回原始尺度（播放量）
    pred_views = np.expm1(log_pred)
    pred_views_lower = np.expm1(log_pred_lower)
    pred_views_upper = np.expm1(log_pred_upper)
    
    # 输出结果
    print("\n" + "-" * 80)
    print("Prediction Results:")
    print("-" * 80)
    print(f"Log scale (log_views):")
    print(f"  Point prediction: {log_pred:.4f}")
    print(f"  95% prediction interval: [{log_pred_lower:.4f}, {log_pred_upper:.4f}]")
    print(f"\nOriginal scale (views):")
    print(f"  Point prediction: {pred_views:,.0f} views")
    print(f"  95% prediction interval: [{pred_views_lower:,.0f}, {pred_views_upper:,.0f}] views")
    print(f"  Interval width: {pred_views_upper - pred_views_lower:,.0f} views")
    
    # 可视化预测结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制预测区间
    ax.barh([0], [pred_views_upper - pred_views_lower], 
            left=pred_views_lower, height=0.3, 
            alpha=0.3, color='steelblue', label='95% Prediction Interval')
    ax.plot([pred_views], [0], 'ro', markersize=12, label='Point Prediction', zorder=5)
    ax.errorbar([pred_views], [0], 
                xerr=[[pred_views - pred_views_lower], [pred_views_upper - pred_views]], 
                fmt='none', ecolor='red', elinewidth=2, capsize=10, capthick=2, zorder=4)
    
    ax.set_xlabel('Predicted Views', fontsize=12)
    ax.set_yticks([0])
    ax.set_yticklabels([scenario.replace('_', ' ').title()])
    title_text = f'Views Prediction Result\n(Scenario: {scenario.replace("_", " ").title()})'
    ax.set_title(title_text, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best')
    
    # 添加数值标签
    ax.text(pred_views, 0.15, f'{pred_views:,.0f}', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(pred_views_lower, -0.2, f'{pred_views_lower:,.0f}', 
           ha='center', va='top', fontsize=9, color='blue')
    ax.text(pred_views_upper, -0.2, f'{pred_views_upper:,.0f}', 
           ha='center', va='top', fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    print("\n✓ Prediction result plot saved as 'prediction_result.png'")
    plt.show()
    
    return {
        'log_pred': log_pred,
        'log_pred_lower': log_pred_lower,
        'log_pred_upper': log_pred_upper,
        'pred_views': pred_views,
        'pred_views_lower': pred_views_lower,
        'pred_views_upper': pred_views_upper,
        'scenario': scenario
    }


# ============================================================================
# 预测对比可视化
# ============================================================================

def visualize_prediction_comparison(prediction_results):
    """
    可视化不同场景下的预测结果对比
    
    Parameters:
    -----------
    prediction_results : list
        包含多个预测结果的列表
    """
    print("\nGenerating prediction comparison visualization...")
    
    scenarios = [r['scenario'] for r in prediction_results]
    pred_views = [r['pred_views'] for r in prediction_results]
    pred_lower = [r['pred_views_lower'] for r in prediction_results]
    pred_upper = [r['pred_views_upper'] for r in prediction_results]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制预测区间
    x_pos = np.arange(len(scenarios))
    width = 0.6
    
    # 计算区间高度
    interval_heights = [u - l for u, l in zip(pred_upper, pred_lower)]
    interval_bottoms = pred_lower
    
    # 绘制区间条形图
    bars = ax.barh(x_pos, interval_heights, left=interval_bottoms, 
                   height=width, alpha=0.3, color='steelblue', 
                   label='95% Prediction Interval')
    
    # 绘制点预测
    ax.scatter(pred_views, x_pos, color='red', s=200, zorder=5, 
              label='Point Prediction', marker='o', edgecolors='black', linewidths=2)
    
    # 添加误差棒
    errors_lower = [p - l for p, l in zip(pred_views, pred_lower)]
    errors_upper = [u - p for u, p in zip(pred_upper, pred_views)]
    ax.errorbar(pred_views, x_pos, 
                xerr=[errors_lower, errors_upper],
                fmt='none', ecolor='red', elinewidth=2, capsize=8, capthick=2, zorder=4)
    
    # 设置标签
    ax.set_yticks(x_pos)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios], fontsize=12)
    ax.set_xlabel('Predicted Views', fontsize=13, fontweight='bold')
    ax.set_title('Prediction Comparison: Different Scenarios', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=11)
    
    # 添加数值标签
    for i, (pred, lower, upper) in enumerate(zip(pred_views, pred_lower, pred_upper)):
        ax.text(pred, i, f'  {pred:,.0f}', ha='left', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(lower, i, f'{lower:,.0f}  ', ha='right', va='center', 
               fontsize=9, color='blue', alpha=0.7)
        ax.text(upper, i, f'  {upper:,.0f}', ha='left', va='center', 
               fontsize=9, color='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction comparison plot saved as 'prediction_comparison.png'")
    plt.show()


# ============================================================================
# 主函数：执行完整分析流程
# ============================================================================

def main():
    """
    主函数：执行完整的回归分析流程
    """
    print("\n" + "=" * 80)
    print("Hurdle Model - Stage 2: Regression Model Analysis")
    print("=" * 80)
    print("\nThis script will execute the following steps:")
    print("  1. Data loading and preprocessing")
    print("  2. Variable transformation and feature engineering (Log transformation)")
    print("  3. Building OLS model")
    print("  4. Model diagnostics (residual analysis, heteroscedasticity test, robust SE)")
    print("  5. Prediction and interval estimation")
    print("=" * 80)
    
    # 数据文件路径
    data_file = 'New_Youtube_Videos_2022_Transformed.csv'
    
    try:
        # Step 0: 加载数据
        df = load_and_prepare_data(data_file)
        
        # Step 1: 特征工程
        df = feature_engineering(df)
        
        # Step 2: 建立 OLS 模型
        model, formula, df = build_ols_model(df)
        
        # Step 3: 模型诊断
        robust_model = model_diagnostics(model, df)
        
        # 如果存在异方差，使用稳健模型进行预测
        final_model = robust_model if robust_model is not None else model
        
        # 保存最终模型的系数（用于原型系统）
        # 如果使用了稳健标准误，保存稳健模型的系数
        print("\n" + "=" * 80)
        print("Saving Final Model Coefficients for Prototype System")
        print("=" * 80)
        save_model_coefficients(final_model, formula)
        
        # Step 4: 预测与区间估计
        # 示例1：高潜力场景
        pred_result1 = prediction_with_intervals(final_model, df, scenario='high_potential')
        
        # 示例2：典型场景
        print("\n" + "=" * 80)
        pred_result2 = prediction_with_intervals(final_model, df, scenario='typical')
        
        # 示例3：低潜力场景（对比）
        print("\n" + "=" * 80)
        pred_result3 = prediction_with_intervals(final_model, df, scenario='low_potential')
        
        # 对比分析
        print("\n" + "=" * 80)
        print("Comparison Analysis: Different Scenarios")
        print("=" * 80)
        print(f"{'Metric':<30} {'High Potential':<20} {'Typical':<20} {'Low Potential':<20}")
        print("-" * 90)
        print(f"{'Predicted Views':<30} {pred_result1['pred_views']:<20,.0f} {pred_result2['pred_views']:<20,.0f} "
              f"{pred_result3['pred_views']:<20,.0f}")
        print(f"{'95% CI Lower':<30} {pred_result1['pred_views_lower']:<20,.0f} {pred_result2['pred_views_lower']:<20,.0f} "
              f"{pred_result3['pred_views_lower']:<20,.0f}")
        print(f"{'95% CI Upper':<30} {pred_result1['pred_views_upper']:<20,.0f} {pred_result2['pred_views_upper']:<20,.0f} "
              f"{pred_result3['pred_views_upper']:<20,.0f}")
        
        # 可视化预测对比
        visualize_prediction_comparison([pred_result1, pred_result2, pred_result3])
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print("\nGenerated files:")
        print("\nGenerated files:")
        print("  📊 Data Files (for prototype system):")
        print("     - model_coefficients.json: Model coefficients (JSON format)")
        print("     - model_coefficients.csv: Model coefficients (CSV format, human-readable)")
        print("  📈 Visualization Files:")
        print("     - model_diagnostics.png: Model diagnostic plots (residual analysis, Q-Q plots, etc.)")
        print("     - coefficient_importance.png: Coefficient importance visualization")
        print("     - actual_vs_predicted.png: Actual vs predicted values comparison")
        print("     - prediction_result.png: Prediction result plot (with prediction intervals)")
        print("     - prediction_comparison.png: Comparison of different prediction scenarios")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file '{data_file}' not found")
        print("   Please ensure the data file is in the current directory")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
