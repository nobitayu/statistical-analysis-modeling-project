## 回归模型数据处理

对于原始数据，我们进行了以下步骤，完成统计预处理以及特征工程，确保各特征适用于建立多元线性回归模型

### 数据筛选与采样

输入 : kaggle获取的原始大数据集

操作 :

- 时间筛选 : 仅保留 publishedAt 在 2022年的视频。
- 趋势筛选 : 仅保留进入趋势榜（ is_trending = 1 ）的视频。
- 采样 : 随机抽取了 5000 条样本。
- 缺失值：清除不开放评论的数据（约占1.1%），因为评论数是进行回归建模的重要特征

### 基础特征工程

新增特征 :
- 时间特征 :
  - period_* : 将发布时间划分为 Dawn/Morning/Afternoon/Evening 并进行 One-Hot 编码。
  - is_weekend : 是否在周末发布 (Bool)。
- 分类特征 :
  - category_* : 对 categoryId 进行 One-Hot 编码。
- 交互特征 :
  - like_rate : 点赞率 ( likes / view_count )。
  - comment_rate : 评论率 ( comment_count / view_count )。
- 标题特征 :
  - title_length : 标题长度。
  - title_upper_ratio : 标题大写字母占比。
  - title_upper_count : 标题大写字母占比。
  - title_has_punct : 标题是否包含感叹号或问号。

### 进阶特征工程

新增特征 :
- 频道特征 :
  - channel_activity : 该频道在样本中的出现频次。
  - channel_avg_views/like_rate/comment_count: 该频道的平均表现数据。
  - channel_name_len : 频道名称长度。
- 内容特征 :
  - tags_count : 视频标签数量。
  - desc_length : 视频描述长度。
  - desc_has_timestamp：描述中是否包含时间戳
  -   desc_keyword_count：描述中包含特定关键词数量

### 数据变换与正态化

操作 :

- 对各个计数/连续性数据进行分布诊断
- 对数变换 : 针对长尾分布的连续变量，应用 ln(x + 1) 变换。
- 目的 : 降低数据偏度，使其更接近正态分布，以满足线性回归等模型的假设。

### 多重共线性检验

操作 : 对最终数据集进行 VIF（方差膨胀因子）分析，检查多重共线性问题。对高度相关变量进行处理，删除或融合。最终得到的VIF Top20如下，均为轻度多重共线性，可接受：

=== VIF 分析结果 (Top 20) ===
                          feature       VIF
31                      log_likes  4.081344
19                   comment_rate  3.796886
37  log_channel_avg_comment_count  3.782744
25                    tag_density  3.490126
34                 log_tags_count  3.459574
32              log_comment_count  2.971702
36          log_channel_avg_views  2.874038
27                    desc_length  2.528419
39                log_desc_length  2.331898
18                      like_rate  2.294291
33               log_title_length  2.031526
7                     category_20  1.766484
3                     category_10  1.727425
5                     category_17  1.664248
30             desc_keyword_count  1.507847
8                     category_22  1.401316
35           log_channel_activity  1.386836
10                    category_25  1.368779
15                    period_Dawn  1.366665
16                 period_Evening  1.301502

### 最终得到的特征数据集如下：

| **类别**                                                    | **特征**                      | **说明 / 用途**                                              |
| ----------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| **1. 基础元数据 (Metadata)**                                | title                         | 视频标题                                                     |
|                                                             | publishedAt                   | 视频发布时间 (yyyy/mm/dd HH:MM)                              |
|                                                             | categoryId                    | 视频分类ID (数字)                                            |
|                                                             | trending_date                 | 视频进入趋势榜的日期                                         |
|                                                             | tags                          | 视频标签 (原始字符串，用                                     |
|                                                             | view_count                    | 视频播放量 [核心因变量]                                      |
|                                                             | likes                         | 点赞数                                                       |
|                                                             | comment_count                 | 评论数                                                       |
| **2. 分类特征 (Categorical Features)**                      | category_1 ~ category_29      | categoryId 独热编码 (One-Hot)，category_24占比最大，作为参照组删除，避免共线性，类似虚拟变量的处理 |
| **3. 时间特征 (Temporal Features)**                         | period_Dawn                   | 是否在凌晨 (0-6点) 发布                                      |
|                                                             | period_Morning                | 是否在上午 (6-12点) 发布                                     |
|                                                             | period_Afternoon              | 是否在下午 (12-18点) 发布，占比最大，作为参照组删除，避免共线性，类似虚拟变量的处理 |
|                                                             | period_Evening                | 是否在晚上 (18-24点) 发布                                    |
|                                                             | is_weekend                    | 是否在周末 (周六/周日) 发布，1=是，0=否                      |
| **4. 互动与标题特征 (Interaction & Title Features)**        | like_rate                     | 点赞率 = likes / view_count                                  |
|                                                             | comment_rate                  | 评论率 = comment_count / view_count                          |
|                                                             | title_length                  | 标题字符长度                                                 |
|                                                             | title_upper_ratio             | 标题中大写字母占比                                           |
|                                                             | title_has_punct               | 标题是否包含感叹号(!)或问号(?)                               |
| **5. 频道与文本衍生特征 (Channel & Text Derived Features)** | channel_activity              | 频道活跃度，数据集中该频道的视频总数                         |
|                                                             | channel_avg_views             | 频道平均播放量                                               |
|                                                             | channel_avg_like_rate         | 频道平均点赞率                                               |
|                                                             | channel_avg_comment_count     | 频道平均评论数                                               |
|                                                             | channel_name_len              | 频道名称长度                                                 |
|                                                             | channel_has_digit             | 频道名称是否包含数字                                         |
|                                                             | channel_has_special           | 频道名称是否包含特殊符号                                     |
|                                                             | tags_count                    | 视频标签数量                                                 |
|                                                             | tag_density                   | 标签密度 (如标签总长度 / 描述长度)                           |
|                                                             | popular_tag_ratio             | 热门标签占比                                                 |
|                                                             | desc_length                   | 视频描述长度                                                 |
|                                                             | desc_has_youtube_link         | 描述中是否包含 YouTube 链接                                  |
|                                                             | desc_has_timestamp            | 描述中是否包含时间戳 (如 "0:00")                             |
|                                                             | desc_keyword_count            | 描述中包含特定关键词数量 (如 "subscribe")                    |
| **6. 对数变换特征 (Log Transformed Features)**              | log_view_count                | ln(view_count + 1)，回归模型因变量 Y                         |
|                                                             | log_likes                     | ln(likes + 1)                                                |
|                                                             | log_comment_count             | ln(comment_count + 1)                                        |
|                                                             | log_title_length              | ln(title_length + 1)                                         |
|                                                             | log_tags_count                | ln(tags_count + 1)                                           |
|                                                             | log_channel_activity          | ln(channel_activity + 1)                                     |
|                                                             | log_channel_avg_views         | ln(channel_avg_views + 1)                                    |
|                                                             | log_channel_avg_comment_count | ln(channel_avg_comment_count + 1)                            |
|                                                             | log_channel_name_len          | ln(channel_name_len + 1)                                     |
|                                                             | log_desc_length               | ln(desc_length + 1)                                          |