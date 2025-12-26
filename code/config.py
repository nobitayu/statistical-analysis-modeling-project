# config.py
import os

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_data.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'output', 'processed_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'ols_model.pkl')

# ================= 特征配置 =================
# 1. 目标变量 (我们会在代码里自动做 log 处理)
TARGET_COL = 'views'

# 2. 数值型特征 (连续变量)
# 只要发布前能拿到的数据
NUMERIC_COLS = [
    'subscriber', 
    'channel_avg_views', 
    'channel_avg_like_rate', 
    'title_length', 
    'tag_density', 
    'desc_length', 
    'desc_keyword_count',
    'title_upper_ratio'
]

# 3. 二值特征 (0/1)
BINARY_COLS = [
    'desc_has_youtube_link',
    'desc_has_timestamp',
    'channel_has_digit',
    'channel_has_special',
    'title_ends_with_punct'
]

# 4. 这里的配置用于逻辑控制，具体列名会在预处理中动态抓取
# 不需要手动列出 category_1...43，代码会自动识别
PREFIX_CATEGORY = 'category_'
PREFIX_PERIOD = 'period_'

# ================= 模型配置 =================
USE_FEATURE_SCALING = True  # 是否使用特征标准化（正则化模型建议开启）
USE_CROSS_VALIDATION = True  # 是否使用交叉验证
CV_FOLDS = 5  # 交叉验证折数
RANDOM_STATE = 42  # 随机种子，保证结果可复现

# ================= 输出路径 =================
PLOTS_DIR = os.path.join(BASE_DIR, 'output', 'plots')
EVALUATION_REPORT_PATH = os.path.join(BASE_DIR, 'output', 'evaluation_report.json')