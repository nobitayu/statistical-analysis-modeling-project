# -*- coding: utf-8 -*-
"""
YouTube视频热度预测原型系统
使用Streamlit构建的Web应用，展示数据分析和模型结果，并提供视频预测功能
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from PIL import Image

# 设置页面配置
st.set_page_config(
    page_title="YouTube视频热度预测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径配置
# 获取当前工作目录，假设从项目根目录运行
import sys
if hasattr(sys, '_getframe'):
    # 尝试从__file__获取路径
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
    except:
        # 如果失败，使用当前工作目录
        BASE_DIR = Path.cwd()
else:
    BASE_DIR = Path.cwd()

# 如果BASE_DIR是system目录，则向上移动一级
if BASE_DIR.name == 'system':
    BASE_DIR = BASE_DIR.parent

MODEL_DIR = BASE_DIR / "model" / "model"
DATA_FILE = BASE_DIR / "process" / "youtube_data_balanced_5000.csv"
MODEL_FILE = MODEL_DIR / "model_pipeline.joblib"
METRICS_FILE = MODEL_DIR / "metrics.txt"
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"
OPTIMIZE_RESULTS_DIR = MODEL_DIR / "optimize_results"
ANALYSIS_IMAGE_FILE = BASE_DIR / "process" / "youtube_analysis_balanced_5000.png"
DISTRIBUTION_ORIGINAL_FILE = BASE_DIR / "process" / "distribution_original.png"
DISTRIBUTION_TRANSFORMED_FILE = BASE_DIR / "process" / "distribution_transformed.png"
TRANSFORMATION_STATS_FILE = BASE_DIR / "process" / "transformation_stats.png"
VIS_BOXPLOT_FILE = BASE_DIR / "process" / "1_log_view_count_boxplot.png"
VIS_CATEGORY_FILE = BASE_DIR / "process" / "2_category_pie_chart.png"
VIS_TAGS_FILE = BASE_DIR / "process" / "3_top_20_tags_bar.png"
VIS_TITLE_UPPER_FILE = BASE_DIR / "process" / "4_upper_ratio_vs_views.png"
VIS_TAG_RATIO_FILE = BASE_DIR / "process" / "5_popular_tag_ratio_vs_views.png"
VIS_TAG_COUNT_FILE = BASE_DIR / "process" / "6_tags_count_vs_views.png"
VIS_PUBLISH_TIME_FILE = BASE_DIR / "process" / "7_time_analysis_views.png"

# 回归模型相关路径
REGRESSION_MODEL_DIR = BASE_DIR / "regression_model"
REGRESSION_COEFFICIENTS_JSON = REGRESSION_MODEL_DIR / "model_coefficients.json"
REGRESSION_COEFFICIENTS_CSV = REGRESSION_MODEL_DIR / "model_coefficients.csv"
REGRESSION_DIAGNOSTICS_IMG = REGRESSION_MODEL_DIR / "model_diagnostics.png"
REGRESSION_COEFF_IMPORTANCE_IMG = REGRESSION_MODEL_DIR / "coefficient_importance.png"
REGRESSION_ACTUAL_VS_PRED_IMG = REGRESSION_MODEL_DIR / "actual_vs_predicted.png"
REGRESSION_PREDICTION_RESULT_IMG = REGRESSION_MODEL_DIR / "prediction_result.png"
REGRESSION_PREDICTION_COMPARISON_IMG = REGRESSION_MODEL_DIR / "prediction_comparison.png"

# 加载数据
@st.cache_data
def load_data():
    """加载数据"""
    try:
        data_path = str(DATA_FILE)
        if not os.path.exists(data_path):
            st.error(f"数据文件不存在: {data_path}")
            st.info(f"当前工作目录: {os.getcwd()}")
            st.info(f"BASE_DIR: {BASE_DIR}")
            st.info(f"尝试的路径: {data_path}")
            return None
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        st.info(f"错误详情: {str(e)}")
        st.info(f"数据文件路径: {DATA_FILE}")
        return None

# 加载模型
@st.cache_resource
def load_model():
    """加载模型"""
    try:
        model_path = str(MODEL_FILE)
        if not os.path.exists(model_path):
            st.error(f"模型文件不存在: {model_path}")
            st.info(f"当前工作目录: {os.getcwd()}")
            st.info(f"BASE_DIR: {BASE_DIR}")
            st.info(f"尝试的路径: {model_path}")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.info(f"错误详情: {str(e)}")
        st.info(f"模型文件路径: {MODEL_FILE}")
        return None

# 加载评估指标
@st.cache_data
def load_metrics():
    """加载评估指标"""
    metrics = {}
    try:
        metrics_path = str(METRICS_FILE)
        if not os.path.exists(metrics_path):
            return {}
        with open(metrics_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line and not line.strip().startswith('='):
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_').replace('_score', '')
                        # 处理ROC AUC的特殊情况
                        if 'roc' in key or 'auc' in key:
                            key = 'auc'
                        try:
                            metrics[key] = float(parts[1].strip())
                        except:
                            pass
        return metrics
    except Exception as e:
        st.warning(f"加载指标失败: {e}")
        return {}

# 加载特征重要性
@st.cache_data
def load_feature_importance():
    """加载特征重要性"""
    try:
        importance_path = str(FEATURE_IMPORTANCE_FILE)
        if not os.path.exists(importance_path):
            return pd.DataFrame()
        df = pd.read_csv(importance_path)
        return df
    except Exception as e:
        st.warning(f"加载特征重要性失败: {e}")
        return pd.DataFrame()

# 加载优化结果
@st.cache_data
def load_optimize_results():
    """加载阈值优化结果"""
    try:
        optimize_dir = str(OPTIMIZE_RESULTS_DIR)
        if not os.path.exists(optimize_dir):
            return None
        # 找到最新的报告文件
        report_files = list(Path(optimize_dir).glob("report_*.json"))
        if report_files:
            latest_report = max(report_files, key=os.path.getmtime)
            with open(str(latest_report), 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"加载优化结果失败: {e}")
    return None

# 加载回归模型系数
@st.cache_data
def load_regression_coefficients():
    """加载回归模型系数"""
    try:
        json_path = str(REGRESSION_COEFFICIENTS_JSON)
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"加载回归模型系数失败: {e}")
    return None

# 加载回归模型系数CSV
@st.cache_data
def load_regression_coefficients_csv():
    """加载回归模型系数CSV"""
    try:
        csv_path = str(REGRESSION_COEFFICIENTS_CSV)
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.warning(f"加载回归模型系数CSV失败: {e}")
    return pd.DataFrame()

# 生成模型预测结果
@st.cache_data
def generate_model_predictions():
    """在测试集上生成预测结果"""
    model = load_model()
    df = load_data()
    
    if model is None or df is None:
        return None, None, None
    
    # 准备数据
    NUMERIC_FEATURES = ['title_length', 'question_exclamation_count', 'tag_count', 'hour', 'tags_in_title', 'is_weekend']
    CATEGORICAL_FEATURES = ['time_period', 'category']
    TARGET = 'is_trending'
    
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)
    
    # 使用与训练时相同的随机种子分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 生成预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return y_test, y_pred, y_prob

# 动态生成图表的函数
def plot_roc_curve_dynamic(y_true, y_prob):
    """动态生成ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title('ROC Curve', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix_dynamic(y_true, y_pred):
    """动态生成混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-trending', 'Trending'])
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=10)
    plt.tight_layout()
    return fig

def plot_roc_curve_with_threshold(y_true, y_prob, threshold):
    """动态生成ROC曲线，并标记当前阈值点"""
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    # 根据当前阈值计算预测结果
    y_pred_thresh = (y_prob >= threshold).astype(int)
    
    # 计算当前阈值对应的FPR和TPR
    tn = ((y_true == 0) & (y_pred_thresh == 0)).sum()
    fp = ((y_true == 0) & (y_pred_thresh == 1)).sum()
    fn = ((y_true == 1) & (y_pred_thresh == 0)).sum()
    tp = ((y_true == 1) & (y_pred_thresh == 1)).sum()
    
    fpr_thresh = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=1.5)
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    ax.plot(fpr_thresh, tpr_thresh, 'ro', markersize=8, label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.set_title('ROC Curve', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_precision_recall_dynamic(y_true, y_prob):
    """动态生成Precision-Recall曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-Recall Curve", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_metrics_vs_threshold_dynamic(y_true, y_prob):
    """动态生成指标随阈值变化的图表"""
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_t, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(thresholds, precisions, label="Precision", linewidth=1.5)
    ax.plot(thresholds, recalls, label="Recall", linewidth=1.5)
    ax.plot(thresholds, f1_scores, label="F1", linewidth=1.5)
    ax.set_xlabel("Threshold", fontsize=9)
    ax.set_ylabel("Metric", fontsize=9)
    ax.set_title("Metrics vs Threshold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# 执行统计检验的函数
def perform_statistical_tests(df):
    """执行统计检验并返回结果"""
    results = []
    
    # 1. 标题长度 vs 是否热门
    try:
        trending_title = df[df['is_trending'] == 1]['title_length']
        non_trending_title = df[df['is_trending'] == 0]['title_length']
        t_stat, p_value = stats.ttest_ind(trending_title, non_trending_title, equal_var=False)
        trending_mean = trending_title.mean()
        non_trending_mean = non_trending_title.mean()
        results.append({
            '测试': '标题长度 vs 是否热门',
            'H₀': '热门和非热门视频的标题长度无显著差异',
            '热门平均': f"{trending_mean:.2f}",
            '非热门平均': f"{non_trending_mean:.2f}",
            '差异': f"{trending_mean - non_trending_mean:.2f}",
            't统计量': f"{t_stat:.4f}",
            'P值': f"{p_value:.4e}",
            '结论': '拒绝H₀，标题长度与是否热门有显著差异 (p < 0.05)' if p_value < 0.05 else '不能拒绝H₀，标题长度与是否热门无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '标题长度 vs 是否热门',
            '错误': str(e)
        })
    
    # 2. 问号感叹号数量 vs 是否热门
    try:
        trending_qe = df[df['is_trending'] == 1]['question_exclamation_count']
        non_trending_qe = df[df['is_trending'] == 0]['question_exclamation_count']
        t_stat, p_value = stats.ttest_ind(trending_qe, non_trending_qe, equal_var=False)
        trending_mean = trending_qe.mean()
        non_trending_mean = non_trending_qe.mean()
        results.append({
            '测试': '问号感叹号数量 vs 是否热门',
            'H₀': '热门和非热门视频的问号感叹号数量无显著差异',
            '热门平均': f"{trending_mean:.2f}",
            '非热门平均': f"{non_trending_mean:.2f}",
            '差异': f"{trending_mean - non_trending_mean:.2f}",
            't统计量': f"{t_stat:.4f}",
            'P值': f"{p_value:.4f}",
            '结论': '拒绝H₀，问号感叹号数量与是否热门有显著差异' if p_value < 0.05 else '不能拒绝H₀，问号感叹号数量与是否热门无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '问号感叹号数量 vs 是否热门',
            '错误': str(e)
        })
    
    # 3. 标签数量 vs 是否热门
    try:
        trending_tags = df[df['is_trending'] == 1]['tag_count']
        non_trending_tags = df[df['is_trending'] == 0]['tag_count']
        t_stat, p_value = stats.ttest_ind(trending_tags, non_trending_tags, equal_var=False)
        trending_mean = trending_tags.mean()
        non_trending_mean = non_trending_tags.mean()
        results.append({
            '测试': '标签数量 vs 是否热门',
            'H₀': '热门和非热门视频的标签数量无显著差异',
            '热门平均': f"{trending_mean:.2f}",
            '非热门平均': f"{non_trending_mean:.2f}",
            '差异': f"{trending_mean - non_trending_mean:.2f}",
            't统计量': f"{t_stat:.4f}",
            'P值': f"{p_value:.4e}",
            '结论': '拒绝H₀，标签数量与是否热门有显著差异' if p_value < 0.05 else '不能拒绝H₀，标签数量与是否热门无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '标签数量 vs 是否热门',
            '错误': str(e)
        })
    
    # 4. 标签是否出现在标题里 vs 是否热门 (卡方检验)
    try:
        contingency_table = pd.crosstab(df['tags_in_title'], df['is_trending'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        results.append({
            '测试': '标签是否出现在标题里 vs 是否热门',
            'H₀': '标签是否出现在标题里与是否热门无关',
            '列联表': f"标签在标题中=0且非热门: {contingency_table.loc[0, 0] if 0 in contingency_table.index else 0}, "
                     f"标签在标题中=0且热门: {contingency_table.loc[0, 1] if 0 in contingency_table.index else 0}, "
                     f"标签在标题中=1且非热门: {contingency_table.loc[1, 0] if 1 in contingency_table.index else 0}, "
                     f"标签在标题中=1且热门: {contingency_table.loc[1, 1] if 1 in contingency_table.index else 0}",
            '卡方统计量': f"{chi2:.4f}",
            'P值': f"{p_value:.4e}",
            '结论': '拒绝H₀，标签是否出现在标题里与是否热门有关' if p_value < 0.05 else '不能拒绝H₀，标签是否出现在标题里与是否热门无关'
        })
    except Exception as e:
        results.append({
            '测试': '标签是否出现在标题里 vs 是否热门',
            '错误': str(e)
        })
    
    # 5. 发布时间是否周末 vs 是否热门 (卡方检验)
    try:
        contingency_table = pd.crosstab(df['is_weekend'], df['is_trending'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        results.append({
            '测试': '发布时间是否周末 vs 是否热门',
            'H₀': '周末发布的视频是否为热门视频与工作日无显著差异',
            '列联表': f"工作日且非热门: {contingency_table.loc[0, 0] if 0 in contingency_table.index else 0}, "
                     f"工作日且热门: {contingency_table.loc[0, 1] if 0 in contingency_table.index else 0}, "
                     f"周末且非热门: {contingency_table.loc[1, 0] if 1 in contingency_table.index else 0}, "
                     f"周末且热门: {contingency_table.loc[1, 1] if 1 in contingency_table.index else 0}",
            '卡方统计量': f"{chi2:.4f}",
            'P值': f"{p_value:.4f}",
            '结论': '拒绝H₀，周末发布的视频是否为热门视频与工作日有显著差异' if p_value < 0.05 else '不能拒绝H₀，周末发布的视频是否为热门视频与工作日无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '发布时间是否周末 vs 是否热门',
            '错误': str(e)
        })
    
    # 6. 发布时间段 vs 是否热门 (卡方检验)
    try:
        contingency_table = pd.crosstab(df['time_period'], df['is_trending'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # 生成简化的列联表描述
        time_table_desc = ""
        for period in ['afternoon', 'dawn', 'evening', 'morning']:
            if period in contingency_table.index:
                time_table_desc += f"{period}: 非热门={contingency_table.loc[period, 0]}, 热门={contingency_table.loc[period, 1]}; "
        
        results.append({
            '测试': '发布时间段 vs 是否热门',
            'H₀': '不同时间段发布的视频是否为热门视频无显著差异',
            '列联表': time_table_desc,
            '卡方统计量': f"{chi2:.4f}",
            'P值': f"{p_value:.4e}",
            '结论': '拒绝H₀，不同时间段发布的视频是否为热门视频有显著差异' if p_value < 0.05 else '不能拒绝H₀，不同时间段发布的视频是否为热门视频无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '发布时间段 vs 是否热门',
            '错误': str(e)
        })
    
    # 7. 类别 vs 是否热门 (卡方检验)
    try:
        contingency_table = pd.crosstab(df['category'], df['is_trending'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # 获取前5个类别的分布
        top_categories = df['category'].value_counts().head(5).index
        category_desc = "前5类别: "
        for cat in top_categories:
            if cat in contingency_table.index:
                category_desc += f"{cat}: 非热门={contingency_table.loc[cat, 0]}, 热门={contingency_table.loc[cat, 1]}; "
        
        results.append({
            '测试': '类别 vs 是否热门',
            'H₀': '不同类别的视频是否为热门视频无显著差异',
            '列联表': category_desc,
            '卡方统计量': f"{chi2:.4f}",
            'P值': f"{p_value:.4e}",
            '结论': '拒绝H₀，不同类别的视频是否为热门视频有显著差异' if p_value < 0.05 else '不能拒绝H₀，不同类别的视频是否为热门视频无显著差异'
        })
    except Exception as e:
        results.append({
            '测试': '类别 vs 是否热门',
            '错误': str(e)
        })
    
    return results

# 主应用
def main():
    # 侧边栏导航
    page = st.sidebar.selectbox(
        "选择页面",
        ["首页", "分类数据分析", "回归数据分析", "逻辑回归模型", "回归模型", "视频预测"]
    )
    
    if page == "首页":
        show_home()
    elif page == "分类数据分析":
        show_classification_data_analysis()
    elif page == "回归数据分析":
        show_regression_data_analysis()
    elif page == "逻辑回归模型":
        show_model_evaluation()
    elif page == "回归模型":
        show_regression_model()
    elif page == "视频预测":
        show_prediction()

def show_classification_data_analysis():
    """分类数据分析页面"""
    st.header("分类数据分析")
    
    # 加载数据
    df = load_data()
    if df is None:
        st.error("无法加载数据")
        return
    
    # 创建标签页菜单
    tab1, tab2 = st.tabs(["可视化分析", "统计检验"])
    
    # 标签页1: 可视化分析
    with tab1:
        st.subheader("数据可视化分析")
        
        # 显示数据摘要
        st.write("**数据摘要**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总样本数", len(df))
        with col2:
            trending_count = df['is_trending'].sum()
            st.metric("热门视频数", trending_count)
        with col3:
            non_trending_count = len(df) - trending_count
            st.metric("非热门视频数", non_trending_count)
        with col4:
            st.metric("热门比例", f"{trending_count/len(df)*100:.1f}%")
        
        # 显示可视化图片
        st.markdown("---")
        st.write("**特征可视化分析图**")
        
        # 检查图片文件是否存在
        image_path = str(ANALYSIS_IMAGE_FILE)
        if os.path.exists(image_path):
            # 显示图片
            try:
                # 尝试使用PIL加载图片，这样可以避免某些路径编码问题
                img = Image.open(image_path)
                st.image(img, caption="YouTube视频特征分析 (热门 vs 非热门各2500条)", use_container_width=True)
            except Exception as e:
                st.error(f"加载图片出错: {e}")
                st.info(f"图片路径: {image_path}")
            
            # 图片说明
            st.markdown("""
            **可视化分析说明:**
            
            1. **标题长度 vs 是否热门**: 热门视频标题通常更短
            2. **问号感叹号数量 vs 是否热门**: 两者差异不大
            3. **标签数量 vs 是否热门**: 非热门视频通常有更多标签
            4. **标签是否出现在标题里 vs 是否热门**: 标签出现在标题中的视频很少成为热门
            5. **发布时间是否周末 vs 是否热门**: 周末发布的视频成为热门的比例略高
            6. **发布时间段 vs 是否热门**: 不同时间段的热门视频分布有差异
            7. **类别 vs 是否热门**: 不同类别的热门视频比例有显著差异
            8. **标题长度 vs 标签数量 vs 是否热门**: 热门视频集中于“短标题+多标签”区域
            9. **发布时间小时分布**: 热门视频在不同时间的发布分布
            """)
        else:
            st.warning(f"可视化图片不存在: {image_path}")
            st.info("请确保已运行process.py生成youtube_analysis_balanced_5000.png文件")
    
    # 标签页2: 统计检验
    with tab2:
        st.subheader("统计检验结果")
        st.markdown("对数据中的各个特征进行统计检验，验证它们与视频是否热门之间的关系。")
        
        # 执行统计检验
        with st.spinner("正在执行统计检验..."):
            test_results = perform_statistical_tests(df)
        
        # 显示统计检验结果
        st.markdown("---")
        st.write("**统计检验详细结果**")
        
        for i, result in enumerate(test_results, 1):
            with st.expander(f"检验{i}: {result['测试']}", expanded=i <= 3):
                if '错误' in result:
                    st.error(f"执行检验时出错: {result['错误']}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**原假设(H₀):** {result['H₀']}")
                        if '热门平均' in result:
                            st.markdown(f"**热门视频平均值:** {result['热门平均']}")
                        if '非热门平均' in result:
                            st.markdown(f"**非热门视频平均值:** {result['非热门平均']}")
                        if '差异' in result:
                            st.markdown(f"**差异:** {result['差异']}")
                        if '列联表' in result:
                            st.markdown(f"**列联表数据:** {result['列联表']}")
                    
                    with col2:
                        if 't统计量' in result:
                            st.metric("t统计量", result['t统计量'])
                        if '卡方统计量' in result:
                            st.metric("卡方统计量", result['卡方统计量'])
                        if 'P值' in result:
                            p_value = float(result['P值'].replace('e-', 'e-'))
                            p_display = result['P值']
                            if p_value < 0.05:
                                st.metric("P值", p_display, delta="显著", delta_color="off")
                            else:
                                st.metric("P值", p_display, delta="不显著", delta_color="inverse")
                    
                    # 结论
                    st.markdown(f"**结论:** {result['结论']}")
        
        # 总结表格
        st.markdown("---")
        st.write("**统计检验总结**")
        
        summary_data = []
        for result in test_results:
            if '错误' not in result:
                is_significant = 'P值' in result and float(result['P值'].replace('e-', 'e-')) < 0.05
                summary_data.append({
                    '检验项目': result['测试'],
                    'P值': result['P值'],
                    '是否显著': '是' if is_significant else '否',
                    '结论': '有显著差异' if is_significant else '无显著差异'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

def show_regression_data_analysis():
    """回归数据分析页面"""
    st.header("回归数据分析")
    
    # 分两个子页面：数据处理 和 数据可视化
    tab1, tab2 = st.tabs(["数据处理", "数据可视化"])
    
    # 第一个子页面：数据处理
    with tab1:
        # 定义大字体样式
        def big_text(text):
            st.markdown(f'<div style="font-size: 24px; line-height: 1.6;">{text}</div>', unsafe_allow_html=True)
            
        # 第一部分：数据筛选与采样
        st.subheader("数据筛选与采样")
        st.markdown("""
        <div style="font-size: 24px; line-height: 1.6;">
        <p><b>输入</b>: kaggle获取的原始大数据集</p>
        <p><b>操作</b>:</p>
        <ul>
        <li><b>时间筛选</b>: 仅保留 publishedAt 在 2022年的视频。</li>
        <li><b>趋势筛选</b>: 仅保留进入趋势榜（<code>is_trending = 1</code>）的视频。</li>
        <li><b>采样</b>: 随机抽取了 5000 条样本。</li>
        <li><b>缺失值</b>: 清除不开放评论的数据（约占1.1%），因为评论数是进行回归建模的重要特征。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 第二部分：基础特征工程
        st.subheader("基础特征工程")
        st.markdown("""
        <div style="font-size: 24px; line-height: 1.6;">
        <p><b>新增特征</b>:</p>
        <ul>
        <li><b>时间特征</b>:
        <ul>
        <li><code>period_*</code>: 将发布时间划分为 Dawn/Morning/Afternoon/Evening 并进行 One-Hot 编码。</li>
        <li><code>is_weekend</code>: 是否在周末发布 (Bool)。</li>
        </ul>
        </li>
        <li><b>分类特征</b>:
        <ul>
        <li><code>category_*</code>: 对 categoryId 进行 One-Hot 编码。</li>
        </ul>
        </li>
        <li><b>交互特征</b>:
        <ul>
        <li><code>like_rate</code>: 点赞率 (<code>likes / view_count</code>)。</li>
        <li><code>comment_rate</code>: 评论率 (<code>comment_count / view_count</code>)。</li>
        </ul>
        </li>
        <li><b>标题特征</b>:
        <ul>
        <li><code>title_length</code>: 标题长度。</li>
        <li><code>title_upper_ratio</code>: 标题大写字母占比。</li>
        <li><code>title_has_punct</code>: 标题是否包含感叹号或问号。</li>
        </ul>
        </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        # 第三部分：进阶特征工程
        st.subheader("进阶特征工程")
        st.markdown("""
        <div style="font-size: 24px; line-height: 1.6;">
        <p><b>新增特征</b>:</p>
        <ul>
        <li><b>频道特征</b>:
        <ul>
        <li><code>channel_activity</code>: 该频道在样本中的出现频次。</li>
        <li><code>channel_avg_views/like_rate/comment_count</code>: 该频道的平均表现数据。</li>
        <li><code>channel_name_len</code>: 频道名称长度。</li>
        <li><code>channel_has_digit</code>: 频道名称是否包含数字。</li>
        <li><code>channel_has_special</code>: 频道名称是否包含特殊符号。</li>
        </ul>
        </li>
        <li><b>标签特征</b>:
        <ul>
        <li><code>tags_count</code>: 视频标签数量。</li>
        <li><code>tag_density</code>: 标签密度 (如标签总长度 / 描述长度)。</li>
        <li><code>popular_tag_ratio</code>: 热门标签占比。</li>
        </ul>
        </li>
        <li><b>描述特征</b>:
        <ul>
        <li><code>desc_length</code>: 视频描述长度。</li>
        <li><code>desc_has_youtube_link</code>: 描述中是否包含 YouTube 链接。</li>
        <li><code>desc_has_timestamp</code>: 描述中是否包含时间戳。</li>
        <li><code>desc_keyword_count</code>: 描述中包含特定关键词数量。</li>
        </ul>
        </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # 第四部分：数据变换与正态化
        st.subheader("数据变换与正态化")
        
        # 图一：原始数据分布
        image_path_original = str(DISTRIBUTION_ORIGINAL_FILE)
        if os.path.exists(image_path_original):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_original, caption="原始数据分布（直方图与Q-Q图）", use_container_width=True)
        else:
            st.info("原始数据分布图 (distribution_original.png) 未找到，请将图片放置在 process 目录下。")
            
        big_text("""
        我们通过计算描述性统计指标发现，部分原始数据的偏度（Skewness）极高，呈现典型的“右偏”或“长尾分布”，这意味着极少数头部视频占据了绝大部分流量；同时，极高的峰度（Kurtosis）也揭示了大量极端离群值的存在。 
        
        我们为呈现偏态的属性绘制直方图和Q-Q图，发现其直方图呈现出左侧陡峭、右侧长尾的形态，而 Q-Q 图中的散点严重偏离对角红线。针对这一问题，我们对具有严重偏态的数据进行对数变换: $y=\ln(x+1)$
        """)
        
        st.markdown("---")
        
        # 图二：变换后数据分布
        image_path_transformed = str(DISTRIBUTION_TRANSFORMED_FILE)
        if os.path.exists(image_path_transformed):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_transformed, caption="变换后数据分布（直方图与Q-Q图）", use_container_width=True)
        else:
            st.info("变换后数据分布图 (distribution_transformed.png) 未找到，请将图片放置在 process 目录下。")
            
        big_text("""
        变换后的 `log_view_count` 等指标在直方图上已呈现明显的正态特征，Q-Q 图中的点也开始紧密贴合红线，证明分布已得到显著矫正。
        """)
        
        big_text("""
        呈现显著偏态的特征进行对数变换前后的偏度、峰度、W值（Shapiro–Wilk正态检验）的相关数据如下:
        """)
        
        # 变换统计数据表图
        image_path_stats = str(TRANSFORMATION_STATS_FILE)
        if os.path.exists(image_path_stats):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_stats, caption="变换统计数据表", use_container_width=True)
        else:
            st.info("统计数据图片 (transformation_stats.png) 未找到，已显示为交互式表格。请将图片放置在 process 目录下以显示图片。")
        
        # 第五部分：多重共线性检测
        st.subheader("多重共线性检测")
        
        big_text("""
        为了防止特征间具有高度相关性发生过拟合，我们引入了方差膨胀因子作为核心检测指标。在判读标准上，通常认为 VIF 小于 5 是安全范围，而一旦超过 10 则存在严重共线性风险。对于独热编码的特征，我们从每一组编码特征中主动移除一列作为基准列。中高度相关的特征，我们将其删除或转化。处理后对所有特征计算VIF，结果VIF为前五名的特征如下：
        """)

        vif_data = [
            {"排名": 1, "特征名称 (Feature)": "log_likes (点赞数对数)", "VIF 值": 4.18, "共线性程度": "低度相关"},
            {"排名": 2, "特征名称 (Feature)": "comment_rate (评论率)", "VIF 值": 3.80, "共线性程度": "低度相关"},
            {"排名": 3, "特征名称 (Feature)": "log_channel_avg_comment_count (频道平均评论数对数)", "VIF 值": 3.79, "共线性程度": "低度相关"},
            {"排名": 4, "特征名称 (Feature)": "tag_density (标签密度)", "VIF 值": 3.49, "共线性程度": "低度相关"},
            {"排名": 5, "特征名称 (Feature)": "log_tags_count (标签数量对数)", "VIF 值": 3.47, "共线性程度": "低度相关"}
        ]
        st.dataframe(pd.DataFrame(vif_data), use_container_width=True, hide_index=True)

        big_text("结果显示可发现特征均为低度相关，可以用于后续建模")

    # 第二个子页面：数据可视化
    with tab2:
        st.subheader("数据可视化分析")
        
        # 定义大字体样式
        def big_text(text):
            st.markdown(f'<div style="font-size: 24px; line-height: 1.6;">{text}</div>', unsafe_allow_html=True)

        # 图1
        image_path_boxplot = str(VIS_BOXPLOT_FILE)
        if os.path.exists(image_path_boxplot):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_boxplot, use_container_width=True)
            big_text("1. 播放量经对数变换后已接近正态分布，其分布中位数约为 13.7，且在高位仍存在部分离群值。即便在热门视频内部，流量分布依然遵循一定的规律，极少数头部视频贡献了极高的观测值。")
        else:
            st.info(f"图1文件未找到: {VIS_BOXPLOT_FILE.name}，请将图片放置在 process 目录下。")

        st.markdown("---")

        # 图2
        image_path_category = str(VIS_CATEGORY_FILE)
        if os.path.exists(image_path_category):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_category, use_container_width=True)
            big_text("2. 对视频所属类别进行统计， Gaming (22.3%)、Entertainment (20.4%) 和 Music (14.1%) 是占比最高的三大类别，用户可能更喜欢轻松向的视频。同时验证了对类别特征进行独热编码的必要性，因为不同赛道的流量基数存在显著的结构性差异。")
        else:
            st.info(f"图2文件未找到: {VIS_CATEGORY_FILE.name}，请将图片放置在 process 目录下。")

        st.markdown("---")

        # 图3
        image_path_tags = str(VIS_TAGS_FILE)
        if os.path.exists(image_path_tags):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_tags, use_container_width=True)
            big_text("3. 对视频的标签进行统计，在数量榜中funny,minecraft,comedy,challenge,gaming居于前五，与视频类别相互应证，用户偏好幽默、轻松、与游戏有关的视频。")
        else:
            st.info(f"图3文件未找到: {VIS_TAGS_FILE.name}，请将图片放置在 process 目录下。")

        st.markdown("---")

        # 图4
        image_path_title = str(VIS_TITLE_UPPER_FILE)
        if os.path.exists(image_path_title):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_title, use_container_width=True)
            big_text("4. 标题中全大写字母的比例越高，播放量有轻微下降的趋势,因此避免使用全大写标题，适度的首字母或关键词大写可能更受欢迎。")
        else:
            st.info(f"图4文件未找到: {VIS_TITLE_UPPER_FILE.name}，请将图片放置在 process 目录下。")

        st.markdown("---")

        # 图5 & 图6
        col1, col2 = st.columns(2)
        with col1:
            image_path_tag_ratio = str(VIS_TAG_RATIO_FILE)
            if os.path.exists(image_path_tag_ratio):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(image_path_tag_ratio, caption="热门标签占比与播放量", use_container_width=True)
            else:
                st.info(f"文件未找到: {VIS_TAG_RATIO_FILE.name}")
        
        with col2:
            image_path_tag_count = str(VIS_TAG_COUNT_FILE)
            if os.path.exists(image_path_tag_count):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(image_path_tag_count, caption="标签总数与播放量", use_container_width=True)
            else:
                st.info(f"文件未找到: {VIS_TAG_COUNT_FILE.name}")
        
        big_text("5. 热门标签占比与播放量呈现正相关趋势，而标签总数的回归线趋于平缓。表明内容的标签质量比单纯的数量堆砌更具预测力。")

        st.markdown("---")

        # 图7
        image_path_publish = str(VIS_PUBLISH_TIME_FILE)
        if os.path.exists(image_path_publish):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path_publish, use_container_width=True)
            big_text("6. 我们对视频发布时间进行了统计，在发布小时维度，峰值出现在上午 9 点，这个时间点发布的视频平均播放量最高，是一个明显的流量高地。凌晨 3 点也有一个小高峰，可能与跨时区观众或深夜党有关。而上午 10 点之后迅速下滑，在中午到傍晚表现相对平稳。在发布日期维度上，周一是最佳发布日，平均播放量最高，周四紧随其后。周日表现最差，这可能与观众在准备下周工作学习，减少了娱乐时间有关。这表明了特征工程中提取“周末效应”特征的正确性，即用户观看行为受工作日/休息日周期深度影响。")
        else:
            st.info(f"图7文件未找到: {VIS_PUBLISH_TIME_FILE.name}，请将图片放置在 process 目录下。")

def show_home():
    """首页"""
    st.title("YouTube视频热度预测系统")
    st.markdown("---")
    st.header("系统介绍")
    st.markdown("""
    本系统是一个YouTube视频热度预测原型系统，主要功能包括：
    
    ### 主要功能
    
    1. **分类数据分析**
       - 数据可视化分析图展示
       - 统计检验结果展示
       - 特征与目标变量关系分析
    
    2. **回归数据分析**
       - 回归模型的数据分析
       - 回归诊断和假设检验
    
    3. **逻辑回归模型**
       - 模型性能指标展示
       - 可视化图表（ROC曲线、混淆矩阵等）
       - 模型参数解释
       - 阈值优化结果
    
    4. **回归模型**
       - Hurdle Model第二阶段：回归模型分析
       - 模型统计摘要（R²、F统计量等）
       - 模型诊断（残差分析、异方差检验）
       - 系数重要性分析和可视化
       - 预测结果与区间估计（不同场景对比）
       - 完整模型系数表
    
    5. **视频预测**
       - 输入视频特征进行预测
       - 预测概率和结果展示
       - 特征重要性分析
    
    ### 模型信息
    
    - **逻辑回归模型**: 使用结构化特征（数值和类别）进行二分类预测
    - **特征数量**: 26个（6个数值特征 + 20个类别特征）
    - **特征类型**: 仅使用结构化特征（数值和类别），不使用文本特征
    - **评估指标**: Accuracy, Precision, Recall, F1 Score, ROC AUC
    
    ### 使用说明
    
    1. 在左侧导航栏选择不同的页面
    2. 在"分类数据分析"页面查看数据分析和统计检验结果
    3. 在"逻辑回归模型"页面查看模型性能和参数
    4. 在"回归模型"页面查看回归模型结果
    5. 在"视频预测"页面输入视频特征进行预测
    """)
    
    # 显示模型基本信息
    metrics = load_metrics()
    if metrics:
        st.markdown("---")
        st.subheader("模型性能概览")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("准确率", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("精确率", f"{metrics.get('precision', 0):.2%}")
        with col3:
            st.metric("召回率", f"{metrics.get('recall', 0):.2%}")
        with col4:
            st.metric("F1分数", f"{metrics.get('f1', 0):.4f}")
        with col5:
            st.metric("ROC AUC", f"{metrics.get('auc', 0):.4f}")

def show_model_evaluation():
    """逻辑回归模型评估页面"""
    st.header("逻辑回归模型")
    
    # 加载数据
    metrics = load_metrics()
    feature_importance = load_feature_importance()
    optimize_results = load_optimize_results()
    
    # 生成预测结果
    y_test, y_pred, y_prob = generate_model_predictions()
    
    # 创建标签页菜单
    tab1, tab2, tab3, tab4 = st.tabs(["性能指标", "可视化图表", "阈值优化", "特征重要性"])
    
    # 标签页1: 性能指标
    with tab1:
        if metrics or (y_test is not None and y_prob is not None):
            # 如果可以从预测结果计算，优先使用动态计算的值
            y_pred_default = None
            if y_test is not None and y_prob is not None:
                # 使用默认阈值0.5计算指标
                y_pred_default = (y_prob >= 0.5).astype(int)
                current_accuracy = accuracy_score(y_test, y_pred_default)
                current_precision = precision_score(y_test, y_pred_default, zero_division=0)
                current_recall = recall_score(y_test, y_pred_default, zero_division=0)
                current_f1 = f1_score(y_test, y_pred_default, zero_division=0)
                current_auc = roc_auc_score(y_test, y_prob)
            else:
                current_accuracy = metrics.get('accuracy', 0)
                current_precision = metrics.get('precision', 0)
                current_recall = metrics.get('recall', 0)
                current_f1 = metrics.get('f1', 0)
                current_auc = metrics.get('auc', 0)
            
            # 主要指标
            st.write("**核心性能指标**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("准确率", f"{current_accuracy:.4f}")
            with col2:
                st.metric("精确率", f"{current_precision:.4f}")
            with col3:
                st.metric("召回率", f"{current_recall:.4f}")
            with col4:
                st.metric("F1分数", f"{current_f1:.4f}")
            with col5:
                st.metric("ROC AUC", f"{current_auc:.4f}")
            
            # 详细指标表格
            st.markdown("---")
            st.write("**指标详细说明**")
            
            if y_test is not None and y_pred_default is not None:
                cm = confusion_matrix(y_test, y_pred_default)
                tn, fp, fn, tp = cm.ravel()
                
                # 计算更多指标
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                metrics_df = pd.DataFrame({
                    '指标': ['准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 
                            '特异性 (Specificity)', 'F1分数', 'ROC AUC'],
                    '数值': [
                        f"{current_accuracy:.4f}",
                        f"{current_precision:.4f}",
                        f"{current_recall:.4f}",
                        f"{specificity:.4f}",
                        f"{current_f1:.4f}",
                        f"{current_auc:.4f}"
                    ],
                    '说明': [
                        '正确预测的样本占总样本的比例',
                        '预测为正类中真正为正类的比例',
                        '真正的正类中被正确识别的比例',
                        '真正的负类中被正确识别的比例',
                        '精确率和召回率的调和平均数',
                        'ROC曲线下面积，衡量模型整体判别能力'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # 混淆矩阵数值
                st.markdown("---")
                st.write("**混淆矩阵**")
                cm_df = pd.DataFrame({
                    '预测: 非热门': [tn, fn],
                    '预测: 热门': [fp, tp]
                }, index=['实际: 非热门', '实际: 热门'])
                st.dataframe(cm_df, use_container_width=True)
            else:
                # 如果无法计算详细指标，至少显示基本指标表格
                metrics_df = pd.DataFrame({
                    '指标': ['准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 
                            'F1分数', 'ROC AUC'],
                    '数值': [
                        f"{current_accuracy:.4f}",
                        f"{current_precision:.4f}",
                        f"{current_recall:.4f}",
                        f"{current_f1:.4f}",
                        f"{current_auc:.4f}"
                    ],
                    '说明': [
                        '正确预测的样本占总样本的比例',
                        '预测为正类中真正为正类的比例',
                        '真正的正类中被正确识别的比例',
                        '精确率和召回率的调和平均数',
                        'ROC曲线下面积，衡量模型整体判别能力'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.warning("无法加载模型指标")
    
    # 标签页2: 可视化图表
    with tab2:
        if y_test is not None and y_pred is not None and y_prob is not None:
            # 第一部分：分类性能评估
            st.write("**分类性能评估**")
            col1, col2 = st.columns(2)
            with col1:
                fig_roc = plot_roc_curve_dynamic(y_test, y_prob)
                st.pyplot(fig_roc, use_container_width=True)
                plt.close(fig_roc)
                st.caption("ROC曲线：展示模型在不同阈值下的真阳性率(TPR)和假阳性率(FPR)的关系。AUC值越接近1，模型区分能力越强。")
            with col2:
                fig_cm = plot_confusion_matrix_dynamic(y_test, y_pred)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
                st.caption("混淆矩阵：展示模型预测结果与真实标签的对比，包括真阳性(TP)、假阳性(FP)、真阴性(TN)、假阴性(FN)。")
            
            # 第二部分：精确率-召回率分析
            st.markdown("---")
            st.write("**精确率-召回率分析**")
            col1, col2 = st.columns(2)
            with col1:
                fig_pr = plot_precision_recall_dynamic(y_test, y_prob)
                st.pyplot(fig_pr, use_container_width=True)
                plt.close(fig_pr)
                st.caption("Precision-Recall曲线：展示精确率和召回率之间的权衡关系。曲线越靠近右上角，模型性能越好。")
            with col2:
                fig_metrics = plot_metrics_vs_threshold_dynamic(y_test, y_prob)
                st.pyplot(fig_metrics, use_container_width=True)
                plt.close(fig_metrics)
                st.caption("指标随阈值变化：展示不同分类阈值下，精确率、召回率和F1分数的变化趋势，帮助选择最优阈值。")
        else:
            st.warning("无法生成预测结果，请检查模型和数据是否加载成功")
    
    # 标签页3: 阈值优化
    with tab3:
        if y_test is not None and y_prob is not None:
            # 阈值滑块
            col1, col2 = st.columns([2, 1])
            with col1:
                # 获取初始阈值值（如果有优化结果，使用最优阈值，否则使用0.5）
                initial_threshold = 0.5
                if optimize_results:
                    initial_threshold = optimize_results.get('best_threshold', 0.5)
                
                threshold = st.slider(
                    "分类阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(initial_threshold),
                    step=0.01,
                    format="%.2f"
                )
            with col2:
                # 显示最优阈值（如果有）
                if optimize_results:
                    best_threshold = optimize_results.get('best_threshold', 0.5)
                    st.metric("推荐阈值", f"{best_threshold:.3f}")
            
            # 根据当前阈值计算预测结果和指标
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            # 计算实时指标
            current_precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            current_recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            current_f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
            current_accuracy = accuracy_score(y_test, y_pred_thresh)
            
            # 显示实时指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("精确率", f"{current_precision:.4f}")
            with col2:
                st.metric("召回率", f"{current_recall:.4f}")
            with col3:
                st.metric("F1分数", f"{current_f1:.4f}")
            with col4:
                st.metric("准确率", f"{current_accuracy:.4f}")
            
            # 显示混淆矩阵和ROC曲线
            col1, col2 = st.columns(2)
            with col1:
                fig_cm = plot_confusion_matrix_dynamic(y_test, y_pred_thresh)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with col2:
                fig_roc = plot_roc_curve_with_threshold(y_test, y_prob, threshold)
                st.pyplot(fig_roc, use_container_width=True)
                plt.close(fig_roc)
            
            # 如果有优化结果，显示对比表格
            if optimize_results:
                st.markdown("---")
                default_metrics = optimize_results.get('default_threshold_metrics', {})
                best_metrics = optimize_results.get('best_threshold_metrics', {})
                best_threshold = optimize_results.get('best_threshold', 0.5)
                
                comparison_df = pd.DataFrame({
                    '指标': ['Precision', 'Recall', 'F1 Score', 'Accuracy'],
                    '当前阈值': [
                        f"{current_precision:.4f}",
                        f"{current_recall:.4f}",
                        f"{current_f1:.4f}",
                        f"{current_accuracy:.4f}"
                    ],
                    '默认阈值(0.5)': [
                        f"{default_metrics.get('precision', 0):.4f}",
                        f"{default_metrics.get('recall', 0):.4f}",
                        f"{default_metrics.get('f1', 0):.4f}",
                        f"{default_metrics.get('accuracy', 0):.4f}"
                    ],
                    f'最优阈值({best_threshold:.3f})': [
                        f"{best_metrics.get('precision', 0):.4f}",
                        f"{best_metrics.get('recall', 0):.4f}",
                        f"{best_metrics.get('f1', 0):.4f}",
                        f"{best_metrics.get('accuracy', 0):.4f}"
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.warning("无法生成预测结果，请检查模型和数据是否加载成功")
    
    # 标签页4: 特征重要性
    with tab4:
        if not feature_importance.empty:
            # 特征表格和可视化并排
            col1, col2 = st.columns([1, 1])
            
            with col1:
                top_features = feature_importance.head(10)
                st.dataframe(
                    top_features[['feature', 'coefficient']].rename(columns={'feature': '特征', 'coefficient': '系数'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                # 特征重要性可视化
                fig, ax = plt.subplots(figsize=(4, 5))
                top_features_viz = feature_importance.head(12)
                colors = ['#ff6b6b' if x < 0 else '#51cf66' for x in top_features_viz['coefficient']]
                ax.barh(range(len(top_features_viz)), top_features_viz['coefficient'], color=colors)
                ax.set_yticks(range(len(top_features_viz)))
                ax.set_yticklabels([
                    f.replace('num_', '').replace('cat_', '').replace('category_', '').replace('time_period_', '') 
                    for f in top_features_viz['feature']
                ], fontsize=8)
                ax.set_xlabel('系数值', fontsize=9)
                ax.set_title('特征重要性（Top 12）', fontsize=10)
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.warning("无法加载特征重要性")

def show_prediction():
    """视频预测页面"""
    st.header("视频预测")
    st.markdown("输入视频特征信息，系统将预测该视频是否会成为热门视频。")
    
    model = load_model()
    if model is None:
        st.error("无法加载模型")
        return
    
    # 加载优化结果以获取最优阈值
    optimize_results = load_optimize_results()
    optimal_threshold = 0.43
    if optimize_results:
        optimal_threshold = optimize_results.get('best_threshold', 0.43)
    
    # 输入表单
    with st.form("prediction_form"):
        st.subheader("视频特征输入")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title_length = st.number_input("标题长度", min_value=1, max_value=200, value=50, step=1)
            question_exclamation_count = st.number_input("问号和感叹号数量", min_value=0, max_value=20, value=0, step=1)
            tag_count = st.number_input("标签数量", min_value=0, max_value=50, value=5, step=1)
            hour = st.number_input("发布时间（小时）", min_value=0, max_value=23, value=14, step=1)
        
        with col2:
            tags_in_title = st.selectbox("标题中是否包含标签", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
            is_weekend = st.selectbox("是否在周末发布", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
            time_period = st.selectbox("时间段", ["dawn", "morning", "afternoon", "evening", "night"], index=2)
            category = st.selectbox("视频类别", [
                "Autos & Vehicles", "Comedy", "Education", "Entertainment",
                "Film & Animation", "Gaming", "Howto & Style", "Music",
                "News & Politics", "Nonprofits & Activism", "People & Blogs",
                "Pets & Animals", "Science & Technology", "Sports",
                "Travel & Events", "Unknown"
            ], index=5)
        
        submitted = st.form_submit_button("预测", use_container_width=True)
        
        if submitted:
            # 准备输入数据
            input_data = {
                'title_length': title_length,
                'question_exclamation_count': question_exclamation_count,
                'tag_count': tag_count,
                'hour': hour,
                'tags_in_title': tags_in_title,
                'is_weekend': is_weekend,
                'time_period': time_period,
                'category': category
            }
            
            # 进行预测
            try:
                X_new = pd.DataFrame([input_data])
                probs = model.predict_proba(X_new)[:, 1]
                pred = model.predict(X_new)[0]
                
                # 显示预测结果
                st.markdown("---")
                st.subheader("预测结果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    prob_percent = probs[0] * 100
                    st.metric("成为热门的概率", f"{prob_percent:.2f}%")
                
                with col2:
                    pred_label = "热门" if pred == 1 else "非热门"
                    st.metric("预测结果", pred_label)
                
                with col3:
                    threshold_pred = "热门" if probs[0] >= optimal_threshold else "非热门"
                    st.metric(f"预测结果（阈值{optimal_threshold:.2f}）", threshold_pred)
                
                # 概率可视化
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh([0], [probs[0]], color='green' if probs[0] >= optimal_threshold else 'red', alpha=0.6)
                ax.barh([0], [1 - probs[0]], left=[probs[0]], color='gray', alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('概率')
                ax.set_title('成为热门的概率')
                ax.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'最优阈值 ({optimal_threshold:.2f})')
                ax.axvline(x=0.5, color='orange', linestyle='--', label='默认阈值 (0.5)')
                ax.legend()
                ax.set_yticks([])
                st.pyplot(fig)
                plt.close(fig)
                
                # 特征影响分析
                st.subheader("特征影响分析")
                feature_importance = load_feature_importance()
                if not feature_importance.empty:
                    # 计算每个特征对预测的贡献
                    st.write("**输入特征值**")
                    st.json(input_data)
                    
                    st.info("提示：标签数量越多，热门概率越低；游戏类别更容易成为热门。")
                
            except Exception as e:
                st.error(f"预测失败: {e}")

def show_regression_model():
    """回归模型页面"""
    st.header("回归模型分析")
    st.markdown("本页面展示Hurdle Model第二阶段：回归模型（The Quantifier）的分析结果，用于预测潜在热门视频的播放量。")
    
    # 加载回归模型数据
    regression_coeff = load_regression_coefficients()
    regression_coeff_df = load_regression_coefficients_csv()
    
    # 创建标签页菜单
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "模型概览", 
        "模型诊断", 
        "系数分析", 
        "预测结果", 
        "模型系数表",
        "播放量预测"
    ])
    
    # 标签页1: 模型概览
    with tab1:
        st.subheader("模型统计摘要")
        
        if regression_coeff:
            model_info = regression_coeff.get('model_info', {})
            
            # 主要指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                r_squared = model_info.get('r_squared', 0)
                st.metric("R² (决定系数)", f"{r_squared:.4f}")
            with col2:
                adj_r_squared = model_info.get('adj_r_squared', 0)
                st.metric("调整R²", f"{adj_r_squared:.4f}")
            with col3:
                f_statistic = model_info.get('f_statistic', 0)
                st.metric("F统计量", f"{f_statistic:.2f}")
            with col4:
                n_obs = model_info.get('n_observations', 0)
                st.metric("样本数", f"{n_obs:,}")
            
            st.markdown("---")
            
            # 模型信息详细说明
            st.write("**模型详细信息**")
            info_data = {
                '指标': [
                    'R² (决定系数)',
                    '调整R²',
                    'F统计量',
                    'F统计量p值',
                    '样本数',
                    '特征数'
                ],
                '数值': [
                    f"{r_squared:.4f}",
                    f"{adj_r_squared:.4f}",
                    f"{f_statistic:.4f}",
                    f"{model_info.get('f_pvalue', 0):.2e}",
                    f"{n_obs:,}",
                    f"{model_info.get('n_features', 0)}"
                ],
                '说明': [
                    '模型解释的方差比例，越接近1越好',
                    '考虑自由度调整后的R²，更稳健',
                    '整体模型显著性检验统计量',
                    'F检验的p值，<0.05表示模型显著',
                    '用于建模的样本数量',
                    '模型中包含的特征变量数量'
                ]
            }
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            
            # 关键显著变量
            st.markdown("---")
            st.write("**关键显著变量 (p < 0.05)**")
            
            significant_vars = []
            coefficients = regression_coeff.get('coefficients', {})
            for var_name, var_info in coefficients.items():
                if var_info.get('is_significant', False):
                    significant_vars.append({
                        '变量': var_name,
                        '系数': f"{var_info.get('coefficient', 0):.6f}",
                        '标准误': f"{var_info.get('std_err', 0):.6f}",
                        'P值': f"{var_info.get('p_value', 1):.4e}"
                    })
            
            if significant_vars:
                sig_df = pd.DataFrame(significant_vars)
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.info("未找到显著变量")
            
            # 数据预处理信息
            st.markdown("---")
            st.write("**数据预处理信息**")
            st.markdown("""
            - **目标变量**: `views` (播放量)
            - **变换方式**: Log变换 (log1p)，降低偏度，使数据更接近正态分布
            - **原始数据统计**:
              - 均值: 1,988,320.39
              - 中位数: 861,138.50
            - **变换后统计**:
              - 均值: 13.7793
              - 中位数: 13.6660
            - **特征工程**:
              - 14个类别特征 (category_24为参照组)
              - 4个时间特征 (period_Afternoon为参照组)
              - 2个标题特征
              - 3个频道特征
              - 5个文本衍生特征
            """)
        else:
            st.warning("无法加载回归模型系数数据")
    
    # 标签页2: 模型诊断
    with tab2:
        st.subheader("模型诊断结果")
        
        if regression_coeff:
            # 残差正态性检验
            st.write("**残差正态性检验 (Shapiro-Wilk Test)**")
            st.markdown("""
            - **统计量**: 0.8872
            - **P值**: 0.0000
            - **结论**: 残差显著偏离正态分布 (α=0.05)
            
            注意：虽然残差不完全符合正态分布，但在大样本情况下（n=4942），中心极限定理保证了估计量的渐近正态性。
            """)
            
            st.markdown("---")
            
            # 异方差检验
            st.write("**异方差检验 (Breusch-Pagan Test)**")
            st.markdown("""
            - **LM统计量**: 592.9971
            - **LM p值**: 0.0000
            - **F统计量**: 23.9250
            - **F p值**: 0.0000
            - **结论**: ⚠ 检测到异方差性 (p < 0.05)
            
            **处理方式**: 使用稳健标准误 (HC3) 重新估计模型，以获得更可靠的统计推断。
            """)
            
            st.markdown("---")
            
            # 稳健标准误模型对比
            st.write("**稳健标准误模型对比**")
            comparison_data = {
                '指标': ['R²', 'F统计量'],
                '原始模型': ['0.8041', '720.43'],
                '稳健SE模型': ['0.8041', '1404.63']
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.info("""
            **说明**: 
            - 稳健标准误模型保持了相同的R²，但F统计量显著提高
            - 使用稳健标准误后，部分变量的显著性发生变化
            - 最终模型使用稳健标准误进行统计推断
            """)
        
        # 模型诊断图
        st.markdown("---")
        st.write("**模型诊断可视化**")
        
        diagnostics_path = str(REGRESSION_DIAGNOSTICS_IMG)
        if os.path.exists(diagnostics_path):
            st.image(diagnostics_path, caption="模型诊断图：残差分析、Q-Q图等", use_container_width=True)
            st.markdown("""
            **诊断图说明**:
            - **残差vs拟合值**: 检查残差的方差齐性
            - **Q-Q图**: 检查残差的正态性
            - **标准化残差**: 识别异常值
            - **Cook距离**: 识别影响模型的高杠杆点
            """)
        else:
            st.warning(f"模型诊断图不存在: {diagnostics_path}")
    
    # 标签页3: 系数分析
    with tab3:
        st.subheader("系数重要性分析")
        
        # 系数重要性图
        coeff_importance_path = str(REGRESSION_COEFF_IMPORTANCE_IMG)
        if os.path.exists(coeff_importance_path):
            st.image(coeff_importance_path, caption="系数重要性可视化", use_container_width=True)
        else:
            st.warning(f"系数重要性图不存在: {coeff_importance_path}")
        
        st.markdown("---")
        
        # 实际vs预测值图
        actual_vs_pred_path = str(REGRESSION_ACTUAL_VS_PRED_IMG)
        if os.path.exists(actual_vs_pred_path):
            st.image(actual_vs_pred_path, caption="实际值 vs 预测值对比", use_container_width=True)
            st.markdown("""
            **说明**: 
            - 该图展示了模型预测值与实际观测值的对比
            - 点越接近对角线，说明预测越准确
            - R² = 0.8041 表示模型解释了约80.4%的方差
            """)
        else:
            st.warning(f"实际vs预测值图不存在: {actual_vs_pred_path}")
        
        # 关键变量解释
        if regression_coeff:
            st.markdown("---")
            st.write("**关键显著变量解释**")
            
            key_vars_explanation = {
                'log_channel_avg_views': {
                    '系数': '0.9473',
                    '解释': '频道平均播放量（对数）每增加1个单位，视频播放量（对数）平均增加0.9473个单位。这是最重要的预测因子。'
                },
                'log_channel_activity': {
                    '系数': '-0.0936',
                    '解释': '频道活跃度（对数）每增加1个单位，视频播放量（对数）平均减少0.0936个单位。这可能是因为活跃度高的频道视频数量多，单个视频的平均播放量相对较低。'
                },
                'category_29': {
                    '系数': '0.1475',
                    '解释': '属于类别29的视频，播放量（对数）平均比参照组（category_24）高0.1475个单位。'
                },
                'category_19': {
                    '系数': '0.1138',
                    '解释': '属于类别19的视频，播放量（对数）平均比参照组高0.1138个单位。'
                },
                'category_23': {
                    '系数': '0.0592',
                    '解释': '属于类别23的视频，播放量（对数）平均比参照组高0.0592个单位。'
                },
                'category_20': {
                    '系数': '0.0391',
                    '解释': '属于类别20的视频，播放量（对数）平均比参照组高0.0391个单位。'
                },
                'log_desc_length': {
                    '系数': '0.0091',
                    '解释': '描述长度（对数）每增加1个单位，视频播放量（对数）平均增加0.0091个单位。'
                }
            }
            
            for var_name, var_info in key_vars_explanation.items():
                with st.expander(f"变量: {var_name}"):
                    st.markdown(f"**系数**: {var_info['系数']}")
                    st.markdown(f"**解释**: {var_info['解释']}")
    
    # 标签页4: 预测结果
    with tab4:
        st.subheader("预测结果与区间估计")
        
        # 预测结果对比
        st.write("**不同场景下的预测结果对比**")
        
        prediction_comparison_data = {
            '指标': [
                '预测播放量',
                '95%置信区间下限',
                '95%置信区间上限',
                '区间宽度'
            ],
            '高潜力场景': [
                '1,909,071',
                '754,856',
                '4,828,143',
                '4,073,287'
            ],
            '典型场景': [
                '842,152',
                '333,056',
                '2,129,428',
                '1,796,372'
            ],
            '低潜力场景': [
                '420,619',
                '166,223',
                '1,064,354',
                '898,131'
            ]
        }
        pred_comparison_df = pd.DataFrame(prediction_comparison_data)
        st.dataframe(pred_comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **场景说明**:
        - **高潜力场景**: 使用75分位数特征值（高潜力特征）
        - **典型场景**: 使用中位数特征值
        - **低潜力场景**: 使用25分位数特征值（低潜力特征）
        
        **预测区间说明**:
        - 预测区间考虑了模型的不确定性
        - 区间宽度反映了预测的不确定性程度
        - 高潜力场景的预测区间更宽，说明高播放量预测的不确定性更大
        """)
        
        st.markdown("---")
        
        # 预测对比图
        pred_comparison_path = str(REGRESSION_PREDICTION_COMPARISON_IMG)
        if os.path.exists(pred_comparison_path):
            st.image(pred_comparison_path, caption="不同场景预测结果对比", use_container_width=True)
        else:
            st.warning(f"预测对比图不存在: {pred_comparison_path}")
        
        st.markdown("---")
        
        # 预测结果图
        pred_result_path = str(REGRESSION_PREDICTION_RESULT_IMG)
        if os.path.exists(pred_result_path):
            st.image(pred_result_path, caption="预测结果图（含预测区间）", use_container_width=True)
            st.markdown("""
            **说明**: 
            - 该图展示了点预测和95%预测区间
            - 预测区间提供了预测的不确定性范围
            - 在原始尺度（views）上，预测区间通常是不对称的
            """)
        else:
            st.warning(f"预测结果图不存在: {pred_result_path}")
        
        # 对数尺度预测结果
        st.markdown("---")
        st.write("**对数尺度预测结果**")
        
        log_prediction_data = {
            '场景': ['高潜力', '典型', '低潜力'],
            '点预测 (log_views)': ['14.4621', '13.6437', '12.9495'],
            '95%区间下限 (log_views)': ['13.5343', '12.7161', '12.0211'],
            '95%区间上限 (log_views)': ['15.3900', '14.5714', '13.8779']
        }
        log_pred_df = pd.DataFrame(log_prediction_data)
        st.dataframe(log_pred_df, use_container_width=True, hide_index=True)
    
    # 标签页5: 模型系数表
    with tab5:
        st.subheader("完整模型系数表")
        
        if not regression_coeff_df.empty:
            # 显示完整系数表
            st.write("**所有变量系数（稳健标准误模型）**")
            
            # 格式化显示
            display_df = regression_coeff_df.copy()
            display_df.columns = ['变量', '系数', '标准误', 'P值', '是否显著']
            
            # 添加显著性标记
            display_df['显著性'] = display_df['是否显著'].apply(lambda x: '是' if x else '否')
            
            # 重新排列列
            display_df = display_df[['变量', '系数', '标准误', 'P值', '显著性']]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # 统计信息
            total_vars = len(display_df)
            significant_vars = display_df['显著性'].value_counts().get('是', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总变量数", total_vars)
            with col2:
                st.metric("显著变量数 (p<0.05)", significant_vars)
            
            st.markdown("---")
            
            # 变量分类统计
            st.write("**变量分类统计**")
            
            category_vars = len([v for v in display_df['变量'] if 'category' in v])
            period_vars = len([v for v in display_df['变量'] if 'period' in v or v == 'is_weekend'])
            title_vars = len([v for v in display_df['变量'] if 'title' in v])
            channel_vars = len([v for v in display_df['变量'] if 'channel' in v])
            text_vars = len([v for v in display_df['变量'] if 'desc' in v or 'tag' in v])
            
            var_category_data = {
                '变量类别': [
                    '类别特征 (category)',
                    '时间特征 (period, is_weekend)',
                    '标题特征 (title)',
                    '频道特征 (channel)',
                    '文本衍生特征 (desc, tag)',
                    '截距项 (Intercept)'
                ],
                '数量': [
                    category_vars,
                    period_vars,
                    title_vars,
                    channel_vars,
                    text_vars,
                    1
                ]
            }
            var_cat_df = pd.DataFrame(var_category_data)
            st.dataframe(var_cat_df, use_container_width=True, hide_index=True)
        else:
            st.warning("无法加载回归模型系数表")
    
    # 标签页6: 播放量预测
    with tab6:
        st.subheader("视频播放量预测")
        st.markdown("输入视频的元数据特征，系统将使用回归模型预测该视频的播放量。")
        
        if regression_coeff is None:
            st.error("无法加载回归模型系数，请确保模型文件存在。")
        else:
            # 创建输入表单
            with st.form("regression_prediction_form"):
                st.write("**视频特征输入**")
                
                # 分为多个列布局
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 类别特征")
                    # 类别选择（category_24是参照组，不显示）
                    category_options = {
                        "category_1": "类别1",
                        "category_2": "类别2",
                        "category_10": "类别10",
                        "category_15": "类别15",
                        "category_17": "类别17",
                        "category_19": "类别19",
                        "category_20": "类别20",
                        "category_22": "类别22",
                        "category_23": "类别23",
                        "category_25": "类别25",
                        "category_26": "类别26",
                        "category_27": "类别27",
                        "category_28": "类别28",
                        "category_29": "类别29",
                        "category_24 (参照组)": None
                    }
                    selected_category = st.selectbox(
                        "视频类别",
                        options=list(category_options.keys()),
                        index=len(category_options)-1,  # 默认选择参照组
                        help="category_24为参照组，选择其他类别将使用对应的系数"
                    )
                    
                    st.markdown("##### 时间特征")
                    period = st.selectbox(
                        "发布时间段",
                        options=["period_Afternoon (参照组)", "period_Dawn", "period_Morning", "period_Evening"],
                        index=0,
                        help="period_Afternoon为参照组"
                    )
                    
                    is_weekend = st.selectbox(
                        "是否在周末发布",
                        options=[0, 1],
                        format_func=lambda x: "是" if x == 1 else "否",
                        index=0
                    )
                    
                    st.markdown("##### 标题特征")
                    title_upper_ratio = st.slider(
                        "标题大写字母占比",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        help="标题中大写字母的比例（0-1之间）"
                    )
                    
                    title_has_punct = st.selectbox(
                        "标题是否包含标点符号（问号/感叹号）",
                        options=[0, 1],
                        format_func=lambda x: "是" if x == 1 else "否",
                        index=0
                    )
                
                with col2:
                    st.markdown("##### 频道特征")
                    channel_activity = st.number_input(
                        "频道活跃度（该频道在样本中的出现频次）",
                        min_value=1,
                        value=10,
                        step=1,
                        help="频道在数据集中的视频数量"
                    )
                    
                    channel_avg_views = st.number_input(
                        "频道平均播放量",
                        min_value=1,
                        value=1000000,
                        step=10000,
                        help="该频道视频的平均播放量"
                    )
                    
                    channel_avg_comment_count = st.number_input(
                        "频道平均评论数",
                        min_value=0,
                        value=5000,
                        step=100,
                        help="该频道视频的平均评论数。注意：该特征的系数为 -2.570383，值过大会导致预测结果不合理。建议范围：0-10,000"
                    )
                    
                    st.markdown("##### 文本特征")
                    tags_count = st.number_input(
                        "标签数量",
                        min_value=0,
                        value=10,
                        step=1,
                        help="视频标签的数量"
                    )
                    
                    tag_density = st.slider(
                        "标签密度",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="标签密度（0-1之间）"
                    )
                    
                    desc_length = st.number_input(
                        "描述长度（字符数）",
                        min_value=0,
                        value=500,
                        step=10,
                        help="视频描述的文字长度"
                    )
                    
                    desc_has_timestamp = st.selectbox(
                        "描述中是否包含时间戳",
                        options=[0, 1],
                        format_func=lambda x: "是" if x == 1 else "否",
                        index=0
                    )
                    
                    desc_keyword_count = st.number_input(
                        "描述中关键词数量",
                        min_value=0,
                        value=5,
                        step=1,
                        help="描述中包含特定关键词的数量"
                    )
                
                submitted = st.form_submit_button("预测播放量", use_container_width=True)
                
                if submitted:
                    try:
                        # 计算预测值
                        # 模型公式：log_views = intercept + sum(coefficient_i * feature_i)
                        # 然后使用 expm1 转换回原始尺度：views = expm1(log_views)
                        
                        intercept = regression_coeff.get('intercept', 0)
                        coefficients = regression_coeff.get('coefficients', {})
                        
                        # 初始化预测值（从截距开始）
                        log_views_pred = intercept
                        contribution_details = []  # 用于调试和显示
                        
                        # 1. 处理类别特征（虚拟变量）
                        if selected_category != "category_24 (参照组)":
                            category_key = selected_category + "[T.True]"
                            if category_key in coefficients:
                                coef_value = coefficients[category_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                contribution_details.append(f"{category_key}: {coef_value:.6f}")
                        
                        # 2. 处理时间特征（虚拟变量）
                        if period == "period_Dawn":
                            period_key = 'period_Dawn[T.True]'
                            if period_key in coefficients:
                                coef_value = coefficients[period_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"{period_key}: {coef_value:.6f}")
                        elif period == "period_Morning":
                            period_key = 'period_Morning[T.True]'
                            if period_key in coefficients:
                                coef_value = coefficients[period_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"{period_key}: {coef_value:.6f}")
                        elif period == "period_Evening":
                            period_key = 'period_Evening[T.True]'
                            if period_key in coefficients:
                                coef_value = coefficients[period_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"{period_key}: {coef_value:.6f}")
                        # period_Afternoon是参照组，系数为0，不需要处理
                        
                        # 3. 处理是否周末（虚拟变量）
                        if is_weekend == 1:
                            weekend_key = 'is_weekend[T.True]'
                            if weekend_key in coefficients:
                                coef_value = coefficients[weekend_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"{weekend_key}: {coef_value:.6f}")
                        
                        # 4. 处理标题特征
                        # 4.1 标题大写字母占比（连续变量）
                        if 'title_upper_ratio' in coefficients:
                            title_upper_coef = coefficients['title_upper_ratio'].get('coefficient', 0)
                            title_upper_contrib = title_upper_coef * title_upper_ratio
                            log_views_pred += title_upper_contrib
                            if title_upper_contrib != 0:
                                contribution_details.append(f"title_upper_ratio: {title_upper_coef:.6f} * {title_upper_ratio:.2f} = {title_upper_contrib:.6f}")
                        
                        # 4.2 标题是否包含标点符号（虚拟变量）
                        if title_has_punct == 1:
                            punct_key = 'title_has_punct[T.True]'
                            if punct_key in coefficients:
                                coef_value = coefficients[punct_key].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"{punct_key}: {coef_value:.6f}")
                        
                        # 5. 处理频道特征（需要log1p变换）
                        # 5.1 频道活跃度
                        log_channel_activity = np.log1p(channel_activity)
                        if 'log_channel_activity' in coefficients:
                            channel_activity_coef = coefficients['log_channel_activity'].get('coefficient', 0)
                            channel_activity_contrib = channel_activity_coef * log_channel_activity
                            log_views_pred += channel_activity_contrib
                            if channel_activity_contrib != 0:
                                contribution_details.append(f"log_channel_activity: {channel_activity_coef:.6f} * {log_channel_activity:.4f} = {channel_activity_contrib:.6f}")
                        
                        # 5.2 频道平均播放量（最重要的特征）
                        log_channel_avg_views = np.log1p(channel_avg_views)
                        if 'log_channel_avg_views' in coefficients:
                            channel_avg_views_coef = coefficients['log_channel_avg_views'].get('coefficient', 0)
                            channel_avg_views_contrib = channel_avg_views_coef * log_channel_avg_views
                            log_views_pred += channel_avg_views_contrib
                            if channel_avg_views_contrib != 0:
                                contribution_details.append(f"log_channel_avg_views: {channel_avg_views_coef:.6f} * {log_channel_avg_views:.4f} = {channel_avg_views_contrib:.6f}")
                        
                        # 5.3 频道平均评论数（注意：系数为负，值过大会导致预测不合理）
                        log_channel_avg_comment_count = np.log1p(channel_avg_comment_count)
                        if 'log_channel_avg_comment_count' in coefficients:
                            channel_comment_coef = coefficients['log_channel_avg_comment_count'].get('coefficient', 0)
                            channel_comment_contrib = channel_comment_coef * log_channel_avg_comment_count
                            log_views_pred += channel_comment_contrib
                            if channel_comment_contrib != 0:
                                contribution_details.append(f"log_channel_avg_comment_count: {channel_comment_coef:.6f} * {log_channel_avg_comment_count:.4f} = {channel_comment_contrib:.6f}")
                        
                        # 6. 处理文本特征
                        # 6.1 标签数量（需要log1p变换）
                        log_tags_count = np.log1p(tags_count)
                        if 'log_tags_count' in coefficients:
                            tags_count_coef = coefficients['log_tags_count'].get('coefficient', 0)
                            tags_count_contrib = tags_count_coef * log_tags_count
                            log_views_pred += tags_count_contrib
                            if tags_count_contrib != 0:
                                contribution_details.append(f"log_tags_count: {tags_count_coef:.6f} * {log_tags_count:.4f} = {tags_count_contrib:.6f}")
                        
                        # 6.2 标签密度（连续变量）
                        if 'tag_density' in coefficients:
                            tag_density_coef = coefficients['tag_density'].get('coefficient', 0)
                            tag_density_contrib = tag_density_coef * tag_density
                            log_views_pred += tag_density_contrib
                            if tag_density_contrib != 0:
                                contribution_details.append(f"tag_density: {tag_density_coef:.6f} * {tag_density:.2f} = {tag_density_contrib:.6f}")
                        
                        # 6.3 描述长度（需要log1p变换）
                        log_desc_length = np.log1p(desc_length)
                        if 'log_desc_length' in coefficients:
                            desc_length_coef = coefficients['log_desc_length'].get('coefficient', 0)
                            desc_length_contrib = desc_length_coef * log_desc_length
                            log_views_pred += desc_length_contrib
                            if desc_length_contrib != 0:
                                contribution_details.append(f"log_desc_length: {desc_length_coef:.6f} * {log_desc_length:.4f} = {desc_length_contrib:.6f}")
                        
                        # 6.4 描述中是否包含时间戳（虚拟变量）
                        if desc_has_timestamp == 1:
                            # 注意：desc_has_timestamp的键名是"desc_has_timestamp"而不是"desc_has_timestamp[T.True]"
                            if 'desc_has_timestamp' in coefficients:
                                coef_value = coefficients['desc_has_timestamp'].get('coefficient', 0)
                                log_views_pred += coef_value
                                if coef_value != 0:
                                    contribution_details.append(f"desc_has_timestamp: {coef_value:.6f}")
                        
                        # 6.5 描述中关键词数量（连续变量）
                        if 'desc_keyword_count' in coefficients:
                            desc_keyword_coef = coefficients['desc_keyword_count'].get('coefficient', 0)
                            desc_keyword_contrib = desc_keyword_coef * desc_keyword_count
                            log_views_pred += desc_keyword_contrib
                            if desc_keyword_contrib != 0:
                                contribution_details.append(f"desc_keyword_count: {desc_keyword_coef:.6f} * {desc_keyword_count} = {desc_keyword_contrib:.6f}")
                        
                        # 7. 转换回原始尺度
                        # 模型预测的是 log(views+1)，所以使用 expm1 转换：views = expm1(log_views) = exp(log_views) - 1
                        if log_views_pred < 0:
                            st.warning(f"⚠️ 警告：对数预测值为负数 ({log_views_pred:.4f})，这可能导致不合理的预测结果。")
                            st.info("可能的原因：输入的特征值超出了训练数据的范围，或者某些特征值过大（特别是频道平均评论数）。")
                            # 即使为负数，也尝试转换（虽然不合理）
                            views_pred = max(0, np.expm1(log_views_pred))
                        else:
                            views_pred = np.expm1(log_views_pred)
                        
                        # 显示预测结果
                        st.markdown("---")
                        st.subheader("预测结果")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("截距 (Intercept)", f"{intercept:.4f}")
                        with col2:
                            st.metric("预测播放量（对数尺度）", f"{log_views_pred:.4f}")
                        with col3:
                            if views_pred < 0:
                                st.metric("预测播放量", "不合理", delta="负数", delta_color="inverse")
                            else:
                                st.metric("预测播放量", f"{views_pred:,.0f}")
                        
                        # 显示计算过程（调试信息）
                        with st.expander("查看详细计算过程", expanded=False):
                            st.write("**计算步骤:**")
                            st.write(f"1. 起始值（截距）: {intercept:.6f}")
                            if contribution_details:
                                st.write("2. 特征贡献:")
                                for i, detail in enumerate(contribution_details, 1):
                                    st.write(f"   {i+1}. {detail}")
                            st.write(f"3. 最终对数预测值: {log_views_pred:.6f}")
                            if log_views_pred < 0:
                                st.warning(f"⚠️ 对数预测值为负数，expm1({log_views_pred:.6f}) = {np.expm1(log_views_pred):.0f} (不合理)")
                                st.info("**建议**: 检查输入值，特别是 `log_channel_avg_comment_count` 的值可能过大。该特征的系数为 -2.570383，当值较大时会产生很大的负贡献。")
                            else:
                                st.write(f"4. 转换回原始尺度: expm1({log_views_pred:.6f}) = {views_pred:,.0f}")
                            
                            # 显示各特征贡献的汇总
                            st.markdown("---")
                            st.write("**特征贡献汇总:**")
                            positive_contrib = sum([float(detail.split('=')[-1].strip()) for detail in contribution_details if '=' in detail and float(detail.split('=')[-1].strip()) > 0])
                            negative_contrib = sum([float(detail.split('=')[-1].strip()) for detail in contribution_details if '=' in detail and float(detail.split('=')[-1].strip()) < 0])
                            st.write(f"- 正贡献总和: {positive_contrib:.4f}")
                            st.write(f"- 负贡献总和: {negative_contrib:.4f}")
                            st.write(f"- 净贡献: {positive_contrib + negative_contrib:.4f}")
                            st.write(f"- 最终预测值: {intercept:.4f} + {positive_contrib + negative_contrib:.4f} = {log_views_pred:.4f}")
                        
                        # 预测范围（简化版本）
                        st.markdown("---")
                        st.write("**预测范围（近似）**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("95%置信区间下限（近似）", f"{views_pred*0.5:,.0f}")
                        with col2:
                            st.metric("95%置信区间上限（近似）", f"{views_pred*2.0:,.0f}")
                        st.caption("注意：这是简化的预测区间，实际预测区间需要考虑模型的标准误和残差方差")
                        
                        # 可视化预测结果
                        st.markdown("---")
                        st.write("**预测结果可视化**")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.barh([0], [views_pred], color='#51cf66', alpha=0.7, height=0.5)
                        ax.set_xlabel('预测播放量', fontsize=12)
                        ax.set_title('视频播放量预测结果', fontsize=14, fontweight='bold')
                        ax.set_yticks([])
                        ax.set_xlim(0, max(views_pred * 1.2, 1000000))
                        
                        # 添加数值标签
                        ax.text(views_pred, 0, f'{views_pred:,.0f}', 
                               ha='left', va='center', fontsize=11, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # 显示输入特征摘要
                        st.markdown("---")
                        st.write("**输入特征摘要**")
                        
                        feature_summary = {
                            '特征类别': [
                                '类别',
                                '发布时间段',
                                '是否周末',
                                '标题大写占比',
                                '标题含标点',
                                '频道活跃度',
                                '频道平均播放量',
                                '频道平均评论数',
                                '标签数量',
                                '标签密度',
                                '描述长度',
                                '描述含时间戳',
                                '描述关键词数'
                            ],
                            '输入值': [
                                selected_category,
                                period,
                                "是" if is_weekend == 1 else "否",
                                f"{title_upper_ratio:.2f}",
                                "是" if title_has_punct == 1 else "否",
                                f"{channel_activity:,}",
                                f"{channel_avg_views:,}",
                                f"{channel_avg_comment_count:,}",
                                f"{tags_count}",
                                f"{tag_density:.2f}",
                                f"{desc_length}",
                                "是" if desc_has_timestamp == 1 else "否",
                                f"{desc_keyword_count}"
                            ]
                        }
                        feature_summary_df = pd.DataFrame(feature_summary)
                        st.dataframe(feature_summary_df, use_container_width=True, hide_index=True)
                        
                        # 显示重要提示
                        st.info("""
                        **提示**:
                        - 预测结果基于回归模型，R² = 0.8041
                        - 预测值存在不确定性，实际播放量可能因多种因素而有所不同
                        - 模型使用对数变换，预测区间在原始尺度上可能不对称
                        - 最重要的预测因子是频道平均播放量（log_channel_avg_views，系数=0.9473）
                        - **注意**: `log_channel_avg_comment_count` 的系数为 -2.570383（不显著，p=0.081），值过大会导致预测结果不合理
                        - 如果预测值为负数，请检查输入值是否在合理范围内
                        """)
                        
                        # 如果预测值为负数，提供更详细的建议
                        if log_views_pred < 0:
                            st.error("""
                            **预测结果不合理的原因分析**:
                            
                            1. **`log_channel_avg_comment_count` 贡献过大**: 
                               - 系数: -2.570383
                               - 当前值: {:.2f} (对应原始值: {:.0f})
                               - 贡献: {:.2f}
                               - 建议: 降低频道平均评论数的输入值（建议 < 10,000）
                            
                            2. **其他可能原因**:
                               - 输入的特征值超出了训练数据的范围
                               - 某些特征组合在训练数据中不常见
                            
                            **建议**: 使用更接近训练数据中位数的特征值进行预测。
                            """.format(
                                log_channel_avg_comment_count,
                                channel_avg_comment_count,
                                channel_comment_contrib
                            ))
                        
                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    main()