# ========================================================================
# LogisticModel.R
# 爆款视频预测：Logistic 回归模型（含 Train/Test 划分、评估指标、阈值优化）
# 使用 video_data_processed.csv
# ========================================================================

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# -------------------------- 加载必要的包 --------------------------
required_packages <- c("dplyr", "pROC")
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
        library(pkg, character.only = TRUE)
    }
}

# -------------------------- 载入数据 --------------------------
cat("===== 载入处理后的数据 =====\n")

data_path <- "D:/uni_study/5th/R/final/statistic/video_data_processed.csv"
data <- read.csv(data_path, stringsAsFactors = FALSE)

cat("数据维度：", nrow(data), "行 × ", ncol(data), "列\n")

# ------------------------- 数据预处理 -------------------------
cat("===== 数据预处理（模型专用） =====\n")

# 转换因子型变量
data$is_hot_f          <- as.factor(data$is_hot)
data$has_question_f    <- as.factor(data$has_question)
data$has_exclamation_f <- as.factor(data$has_exclamation)
data$is_weekend_f      <- as.factor(data$is_weekend)

# ------------------------- Train/Test 划分 -------------------------
cat("===== Train/Test 划分 (70%/30%) =====\n")

set.seed(42)
n <- nrow(data)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

cat("训练集：", nrow(train_data), "行\n")
cat("测试集：", nrow(test_data), "行\n")

# ------------------------- 构建 Logistic 模型 -------------------------
cat("===== 构建 Logistic 回归模型 =====\n")

log_model <- glm(
    is_hot_f ~ publish_hour + title_length + tags_count +
        has_question_f + has_exclamation_f +
        is_weekend_f + tag_appeared_in_title_count +
        subscriber,
    data = train_data,
    family = binomial(link = "logit")
)

cat("\n===== Logistic 回归模型 Summary =====\n")
print(summary(log_model))

# ------------------------- 计算优势比 OR 及 95% CI -------------------------
cat("\n===== 模型优势比 OR（Odds Ratio）及 95% CI =====\n")

coef_summary <- summary(log_model)$coefficients
OR <- exp(coef(log_model))
CI <- exp(confint(log_model))

interpret_table <- data.frame(
    Variable = names(coef(log_model)),
    Beta = coef(log_model),
    OR = OR,
    OR_Lower = CI[, 1],
    OR_Upper = CI[, 2],
    P_Value = coef_summary[, 4],
    row.names = NULL
)
print(interpret_table)

# ------------------------- 评估指标计算函数 -------------------------
calc_metrics <- function(y_true, prob, thr) {
    pred <- ifelse(prob >= thr, 1, 0)
    TP <- sum(pred == 1 & y_true == 1)
    FP <- sum(pred == 1 & y_true == 0)
    TN <- sum(pred == 0 & y_true == 0)
    FN <- sum(pred == 0 & y_true == 1)
    
    acc  <- (TP + TN) / (TP + TN + FP + FN)
    prec <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    rec  <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    spec <- ifelse(TN + FP == 0, NA, TN / (TN + FP))
    f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0, NA, 2 * prec * rec / (prec + rec))
    
    c(Accuracy = acc, Precision = prec, Recall = rec, Specificity = spec, F1 = f1,
      TP = TP, FP = FP, TN = TN, FN = FN)
}

# ------------------------- 测试集预测与评估 -------------------------
cat("\n===== 测试集预测与评估 =====\n")

test_data$pred_prob <- predict(log_model, newdata = test_data, type = "response")

# ROC & AUC
roc_obj <- roc(test_data$is_hot, test_data$pred_prob)
auc_value <- as.numeric(auc(roc_obj))
cat("AUC 值：", auc_value, "\n")

# 默认阈值 0.2 的指标
metrics_default <- calc_metrics(test_data$is_hot, test_data$pred_prob, 0.2)
cat("\n===== 默认阈值 0.2 的评估指标 =====\n")
print(metrics_default)

# ------------------------- 阈值优化（最大化 F1） -------------------------
cat("\n===== 阈值优化（最大化 F1） =====\n")

grid <- seq(0.05, 0.95, by = 0.01)
metrics_matrix <- t(sapply(grid, function(t) calc_metrics(test_data$is_hot, test_data$pred_prob, t)))
best_idx <- which.max(metrics_matrix[, "F1"])
best_threshold <- grid[best_idx]

cat("最优阈值：", best_threshold, "\n")

metrics_best <- calc_metrics(test_data$is_hot, test_data$pred_prob, best_threshold)
cat("\n===== 最优阈值的评估指标 =====\n")
print(metrics_best)

# ------------------------- 优化前后对比 -------------------------
cat("\n===== 优化前后对比 =====\n")
comparison <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "Specificity", "F1"),
    Default_0.2 = metrics_default[1:5],
    Best_Threshold = metrics_best[1:5]
)
print(comparison)

# ------------------------- 保存 ROC 曲线 -------------------------
cat("\n===== 保存 ROC 曲线 =====\n")

png(filename = "D:/uni_study/5th/R/final/statistic/plots/logistic_ROC.png",
    width = 800, height = 600)
plot(
    roc_obj,
    col = "#f44336",
    lwd = 3,
    main = paste0("Logistic 回归 ROC 曲线（AUC = ", round(auc_value, 4), "）")
)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

cat("\n===== Logistic 回归模型构建完成 ✔ =====\n")
cat("最优阈值：", best_threshold, "\n")
cat("AUC：", auc_value, "\n")
