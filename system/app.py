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
            st.image(image_path, caption="YouTube视频特征分析 (热门 vs 非热门各2500条)", use_container_width=True)
            
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
    
    st.info("回归数据分析部分正在开发中，敬请期待。")
    
    # 预留位置，后续可以添加回归模型的数据分析内容
    # 结构可以参考分类数据分析部分：
    # - 数据分布可视化
    # - 相关性分析
    # - 回归诊断图
    # - 假设检验结果

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