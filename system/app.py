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
       - 回归模型评估和结果展示
       - 模型参数解释
    
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
    st.header("回归模型")
    
    st.info("回归模型部分正在开发中，敬请期待。")
    
    # 预留位置，后续可以添加回归模型的内容
    # 结构可以参考逻辑回归模型部分：
    # - 模型性能指标
    # - 可视化图表
    # - 模型参数解释
    # - 预测功能

if __name__ == "__main__":
    main()