# ==============================================================================
# 视频热度预测数据预处理与可视化脚本
# 功能：数据清洗、特征构造、异常值处理、相关性分析、多维度可视化分析
# 说明：需将数据文件路径替换为实际路径，确保列名匹配
# ==============================================================================

# --------------------------- 安装与加载必要的包 ---------------------------
# 自动检测并安装缺失的包
required_packages <- c("ggplot2", "mice", "dplyr", "lubridate", "ggpubr", "outliers", "corrplot")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# --------------------------- 准备工作 ---------------------------
# 清除环境变量
rm(list = ls())

# 导入数据（请将文件路径替换为你的实际数据路径，如："~/video_data.csv"）
data <- read.csv("E:/tjfx/USvideos_modified.csv", stringsAsFactors = FALSE)

# 初步查看数据结构
cat("===== 数据基本结构 =====\n")
str(data)
cat("\n===== 数据描述性统计 =====\n")
summary(data)

# 缺失值模式可视化
cat("\n===== 缺失值模式 =====\n")
md.pattern(data, rotate.names = TRUE)  

# 检测重复行
dup_count <- length(which(duplicated(data)))
cat(paste0("\n重复行数量：", dup_count, "\n"))

# --------------------------- 数据预处理 ---------------------------
# 1. 保留指定列并初步清理
cat("\n===== 开始数据预处理 =====\n")
# 定义需要保留的原始列
keep_cols <- c("video_id", "publish_date", "publish_hour", "views", "likes", 
               "comment_count", "tag_appeared_in_title_count", "tag_appeared_in_title", 
               "title", "tags_count", "subscriber", "comments_disabled")

# 检查数据中是否存在指定列，过滤出存在的列
existing_cols <- intersect(keep_cols, colnames(data))
data <- data[, existing_cols]
cat(paste0("保留的原始列：", paste(existing_cols, collapse = ", "), "\n"))

if ("comments_disabled" %in% colnames(data)) {
  before_del <- nrow(data)
  
  # 第一步：将字符型转换为逻辑型（关键）
  data <- data %>% 
    mutate(comments_disabled = as.logical(comments_disabled)) %>% 
    # 第二步：按逻辑型筛选（和原代码一致）
    filter(comments_disabled == FALSE)
  
  after_del <- nrow(data)
  cat(paste0("删除comments_disabled为TRUE的行：", before_del - after_del, "行\n"))
}

# 删除含有缺失值的行
before_na <- nrow(data)
data <- na.omit(data)
after_na <- nrow(data)
cat(paste0("删除缺失值行：", before_na - after_na, "行\n"))

# 重置行索引
rownames(data) <- NULL

# 2. 构造新特征
cat("\n===== 构造新特征 =====\n")
# 计算互动率：(点赞数+评论数)/播放量
data <- data %>% 
  mutate(interaction_rate = (likes + comment_count) / views)

# 计算标题长度
data <- data %>% 
  mutate(title_length = nchar(title))

# 标题中是否有问号（1=有，0=无）
data <- data %>% 
  mutate(has_question = ifelse(grepl("\\?", title), 1, 0))

# 标题中是否有感叹号（1=有，0=无）
data <- data %>% 
  mutate(has_exclamation = ifelse(grepl("\\!", title), 1, 0))

# 将publish_date转换为是否为周末（1=周末，0=工作日）
data <- data %>% 
  mutate(publish_date = as.Date(publish_date)) %>% 
  mutate(weekday = wday(publish_date)) %>% 
  mutate(is_weekend = ifelse(weekday %in% c(1, 7), 1, 0))

# 定义爆款：互动率前15%为1，否则为0
threshold <- quantile(data$interaction_rate, 0.85)
data <- data %>% 
  mutate(is_hot = ifelse(interaction_rate >= threshold, 1, 0))

# 保留数值型副本用于相关性分析，同时将分类特征转换为因子类型
data$tag_appeared_in_title_f <- as.factor(data$tag_appeared_in_title)
data$is_weekend_f <- as.factor(data$is_weekend)
data$has_question_f <- as.factor(data$has_question)
data$has_exclamation_f <- as.factor(data$has_exclamation)
data$is_hot_f <- as.factor(data$is_hot)

# 查看构造后的特征
cat("构造后的特征结构：\n")
str(data[, c("interaction_rate", "title_length", "has_question", "is_weekend", "is_hot")])

# 3. 异常值检测与处理（IQR方法）
cat("\n===== 异常值处理 =====\n")
# 定义数值型特征列
numeric_cols <- c("views", "likes", "comment_count", "tag_appeared_in_title_count", 
                  "tags_count", "subscriber", "interaction_rate", "title_length", "publish_hour",
                  "is_weekend", "has_question", "has_exclamation", "is_hot")

# 最终保留的列
final_cols <- c("video_id", "is_weekend", "publish_hour", "views", "likes", 
                "comment_count", "tag_appeared_in_title_count", "tag_appeared_in_title_f", 
                "title_length", "has_question", "has_exclamation", "tags_count", 
                "subscriber", "interaction_rate", "is_hot",
                "is_weekend_f", "has_question_f", "has_exclamation_f", "is_hot_f")
data <- data[, final_cols]
cat(paste0("\n最终数据维度：", nrow(data), "行 × ", ncol(data), "列\n"))

# --------------------------- 新增：保存处理好的数据 ---------------------------
cat("\n===== 保存处理后的数据 =====\n")
# 1. 设置保存路径（可自定义，建议使用绝对路径）
save_dir <- "E:/tjfx/"  # 保存目录
csv_file <- paste0(save_dir, "video_data_processed.csv")  # CSV文件路径

# 2. 检查保存目录是否存在，不存在则创建
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE)  # recursive=TRUE支持创建多级目录
  cat(paste0("创建保存目录：", save_dir, "\n"))
}

# 3. 保存为CSV文件（跨平台通用，Excel/Python可直接打开）
write.csv(
  x = data,
  file = csv_file,
  row.names = FALSE  # 不保存行索引，更整洁
)
cat(paste0("已保存CSV格式数据至：", csv_file, "\n"))

# --------------------------- 相关性分析与热力图可视化 ---------------------------
cat("\n===== 相关性分析与热力图绘制 =====\n")
# 1. 选择用于相关性分析的数值特征（排除因子型和非数值列）
corr_features <- c("publish_hour", "views", "likes", "comment_count", "tag_appeared_in_title_count",
                   "title_length", "tags_count", "subscriber", "interaction_rate")
corr_data <- data[, corr_features]

# 2. 计算相关系数矩阵（Pearson相关系数，也可替换为Spearman）
corr_matrix <- cor(corr_data, method = "pearson")
# 保留两位小数，提升可读性
corr_matrix <- round(corr_matrix, 2)

# 3. 绘制相关性矩阵热力图
# 自定义配色方案（蓝-白-红，对应负相关-无相关-正相关）
color_pal <- colorRampPalette(c("#2196F3", "#FFFFFF", "#F44336"))(100)

# 绘制热力图（上三角，带数值标注）
corrplot(
  corr_matrix,
  method = "color",        # 颜色填充模式
  type = "upper",          # 仅显示上三角，避免重复
  addCoef.col = "black",   # 相关系数文字颜色
  number.cex = 0.7,        # 系数文字大小
  tl.cex = 0.8,            # 变量名文字大小
  tl.srt = 45,             # 变量名旋转45度，避免重叠
  col = color_pal,         # 自定义配色
  bg = "white",
  title = "视频特征相关性矩阵热力图",
  mar = c(0, 0, 2, 0)      # 调整边距（下、左、上、右）
)


# --------------------------- 数据可视化分析 ---------------------------
cat("\n===== 开始数据可视化 =====\n")
# 设置绘图主题
theme_set(theme_minimal(base_size = 10))

# 1. 分类特征与爆款的关系（条形图）
cat("绘制分类特征与爆款的条形图...\n")
# 1.1 是否周末发布 vs 爆款
p1 <- ggplot(data, aes(x = is_weekend_f, fill = is_hot_f)) +
  geom_bar(position = "fill", alpha = 0.7) +
  labs(title = "是否周末发布与爆款的关系", x = "是否周末（0=工作日，1=周末）", 
       y = "比例", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 1.2 标题是否有问号 vs 爆款
p2 <- ggplot(data, aes(x = has_question_f, fill = is_hot_f)) +
  geom_bar(position = "fill", alpha = 0.7) +
  labs(title = "标题是否有问号与爆款的关系", x = "是否有问号（0=无，1=有）", 
       y = "比例", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 1.3 标题是否有感叹号 vs 爆款
p3 <- ggplot(data, aes(x = has_exclamation_f, fill = is_hot_f)) +
  geom_bar(position = "fill", alpha = 0.7) +
  labs(title = "标题是否有感叹号与爆款的关系", x = "是否有感叹号（0=无，1=有）", 
       y = "比例", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 1.4 标签是否出现在标题中 vs 爆款
p4 <- ggplot(data, aes(x = tag_appeared_in_title_f, fill = is_hot_f)) +
  geom_bar(position = "fill", alpha = 0.7) +
  labs(title = "标签是否出现在标题中与爆款的关系", x = "标签是否在标题中", 
       y = "比例", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 组合分类特征图表
categorical_plot <- ggarrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
print(categorical_plot)

# 2. 数值特征与爆款的关系（箱线图）
cat("绘制数值特征与爆款的箱线图...\n")
# 2.1 发布小时 vs 互动率
p5 <- ggplot(data, aes(x = is_hot_f, y = publish_hour, fill = is_hot_f)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "发布小时与爆款的关系", x = "是否爆款", y = "发布小时", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62")) +
  # 关键：设置Y轴刻度为2小时一格
  scale_y_continuous(
    breaks = seq(0, 23, by = 2),  # 从0到23，步长为2生成刻度（0,2,4,...,22）
    limits = c(0, 23)  # 固定Y轴范围为0-23（发布小时的合理范围），可选但推荐
  )

# 2.2 标题长度 vs 互动率
p6 <- ggplot(data, aes(x = is_hot_f, y = title_length, fill = is_hot_f)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "标题长度与爆款的关系", x = "是否爆款", y = "标题长度", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 2.3 标签数量 vs 互动率
p7 <- ggplot(data, aes(x = is_hot_f, y = tags_count, fill = is_hot_f)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "标签数量与爆款的关系", x = "是否爆款", y = "标签数量", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 2.4 订阅数 vs 互动率
p8 <- ggplot(data, aes(x = is_hot_f, y = subscriber, fill = is_hot_f)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "订阅数与爆款的关系", x = "是否爆款", y = "订阅数", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 组合数值特征箱线图
boxplot_plot <- ggarrange(p5, p6, p7, p8, ncol = 2, nrow = 2)
print(boxplot_plot)

# 3. 数值特征间的关系（散点图）
cat("绘制数值特征散点图...\n")
# 3.1 订阅数 vs 互动率（按爆款着色）
p9 <- ggplot(data, aes(x = subscriber, 
                       y = as.numeric(as.character(is_hot_f)),  # 转为数值型（0/1）
                       color = is_hot_f)) +
  geom_jitter(alpha = 0.6, size = 1.5, width = 0.05) +  # width控制x轴抖动幅度
  geom_smooth(method = "lm", se = FALSE, linewidth = 1) +  # 线性趋势线
  labs(title = "订阅数与是否爆款的关系", 
       x = "订阅数", 
       y = "是否爆款（0=非爆款，1=爆款）", 
       color = "是否爆款") +
  scale_color_manual(values = c("#66c2a5", "#fc8d62")) +
  scale_y_continuous(breaks = c(0, 1)) +  # 固定y轴刻度为0和1
  theme_minimal()

# 3.2 标题长度 vs 互动率（按爆款着色）
p10 <- ggplot(data, aes(x = title_length, 
                        y = as.numeric(as.character(is_hot_f)), 
                        color = is_hot_f)) +
  geom_jitter(alpha = 0.6, size = 1.5, width = 0.05) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1) +
  labs(title = "标题长度与是否爆款的关系", 
       x = "标题长度", 
       y = "是否爆款（0=非爆款，1=爆款）", 
       color = "是否爆款") +
  scale_color_manual(values = c("#66c2a5", "#fc8d62")) +
  scale_y_continuous(breaks = c(0, 1)) +
  theme_minimal()

# 3.3 标签数量 vs 互动率（按爆款着色）
p11 <- ggplot(data, aes(x = tags_count, 
                        y = as.numeric(as.character(is_hot_f)), 
                        color = is_hot_f)) +
  geom_jitter(alpha = 0.6, size = 1.5, width = 0.05) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1) +
  labs(title = "标签数量与是否爆款的关系", 
       x = "标签数量", 
       y = "是否爆款（0=非爆款，1=爆款）", 
       color = "是否爆款") +
  scale_color_manual(values = c("#66c2a5", "#fc8d62")) +
  scale_y_continuous(breaks = c(0, 1)) +
  theme_minimal()

# 组合散点图
scatter_plot <- ggarrange(p9, p10, p11, ncol = 2, nrow = 2)
print(scatter_plot)

# 4. 特征分布可视化
cat("绘制特征分布图表...\n")
# 互动率分布（按爆款分组）
p12 <- ggplot(data, aes(x = interaction_rate, fill = is_hot_f)) +
  geom_density(alpha = 0.5) +
  labs(title = "互动率分布", x = "互动率", y = "密度", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

# 发布小时分布
p13 <- ggplot(data, aes(x = factor(publish_hour), fill = is_hot_f)) +
  geom_bar(alpha = 0.7) +
  labs(title = "发布小时分布", x = "发布小时", y = "视频数量", fill = "是否爆款") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 组合分布图表
distribution_plot <- ggarrange(p12, p13, ncol = 2, nrow = 1)
print(distribution_plot)

cat("\n===== 数据预处理、相关性分析与可视化完成 =====\n")