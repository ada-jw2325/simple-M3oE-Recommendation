# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os
import warnings
from config import *

warnings.filterwarnings('ignore')

print("Step 1: Loading raw data...")
# 加载所有需要的CSV文件
train_log_df = pd.read_csv(LOG_STANDARD_TRAIN)
test_log_std_df = pd.read_csv(LOG_STANDARD_TEST)
test_log_rand_df = pd.read_csv(LOG_RANDOM_TEST)
user_features_df = pd.read_csv(USER_FEATURES)
video_basic_df = pd.read_csv(VIDEO_BASIC_FEATURES)
video_stat_df = pd.read_csv(VIDEO_STAT_FEATURES)

print("Renaming columns to prevent conflicts...")


# --- 1. 重命名 user_features_df 中的列 ---
user_features_df.rename(columns={
    'follow_user_num': 'user_following_count', # 用户关注了多少人
    'fans_user_num': 'user_fan_count',         # 用户的粉丝数
    'friend_user_num': 'user_friend_count',    # 用户的好友数
    'register_days': 'user_register_days'      # 用户的注册天数
}, inplace=True)

# --- 2. 重命名 video_features_basic_df 中的列 ---
video_basic_df.rename(columns={
    'video_duration': 'video_total_duration' # 视频的总时长
}, inplace=True)

# --- 3. 重命名 video_features_stat_df 中的列 ---
video_stat_df.columns = ['video_id'] + ['stat_' + col for col in video_stat_df.columns if col != 'video_id']


# 合并测试集
test_log_df = pd.concat([test_log_std_df, test_log_rand_df], ignore_index=True)

# def check(df):
#     return (df['tab'] == 0) | (df['tab'] == 1) | (df['tab'] == 2) | (df['tab'] == 4) | (df['tab'] == 6) | (df['tab'] == 11) | (df['tab'] == 14)  


print("Step 2: Merging dataframes...")
# 以log数据为主表，合并所有特征
train_df = train_log_df.merge(user_features_df, on='user_id', how='left')
train_df = train_df.merge(video_basic_df, on='video_id', how='left')
train_df = train_df.merge(video_stat_df, on='video_id', how='left')

test_df = test_log_df.merge(user_features_df, on='user_id', how='left')
test_df = test_df.merge(video_basic_df, on='video_id', how='left')
test_df = test_df.merge(video_stat_df, on='video_id', how='left')

print("Step 2.1: Cleaning inconsistent data...")
train_df = train_df[~((train_df['is_click'] == 0) & (train_df['long_view'] == 1))]
test_df = test_df[~((test_df['is_click'] == 0) & (test_df['long_view'] == 1))]

# valid_tabs = [0, 1, 2, 4, 6, 11, 14]
valid_tabs = [0, 1, 2, 4, 6]


# 2. 然后直接使用 .isin() 方法进行筛选
train_df = train_df[train_df['tab'].isin(valid_tabs)]
test_df = test_df[test_df['tab'].isin(valid_tabs)]



''' 预处理 时间特征和标签向量
时间特征在这里其实直接看成离散变量即可, 而且在数据集中我发现只有三个日期, 而且这里做不了(和用户曝光时候的差值)

'''

# ==================== 新增的 time_diff 特征工程 ====================
print("Step 2.5: Engineering 'time_diff_minutes' feature...")

# --- 处理训练集 ---
# 1. 将毫秒时间戳和日期字符串都转换为datetime对象
train_df['interaction_time'] = pd.to_datetime(train_df['time_ms'], unit='ms')
train_df['upload_time'] = pd.to_datetime(train_df['upload_dt'], errors='coerce') # errors='coerce' 会将无法解析的日期变为NaT

# 2. 计算两个时间的差值（结果是Timedelta对象）
time_diff = train_df['interaction_time'] - train_df['upload_time']

# 3. 将差值转换为总分钟数，并转换为整数
# .dt.total_seconds() 是获取总秒数的标准方法
train_df['time_diff_minutes'] = (time_diff.dt.total_seconds() / 60).astype('int64')

# --- 处理测试集（使用完全相同的逻辑） ---
test_df['interaction_time'] = pd.to_datetime(test_df['time_ms'], unit='ms')
test_df['upload_time'] = pd.to_datetime(test_df['upload_dt'], errors='coerce')
time_diff_test = test_df['interaction_time'] - test_df['upload_time']
test_df['time_diff_minutes'] = (time_diff_test.dt.total_seconds() / 60).astype('int64')

# --- 清理临时的datetime列 ---
train_df.drop(columns=['interaction_time', 'upload_time'], inplace=True)
test_df.drop(columns=['interaction_time', 'upload_time'], inplace=True)

# 别忘了处理可能因为转换失败产生的缺失值
train_df['time_diff_minutes'].fillna(0, inplace=True)
test_df['time_diff_minutes'].fillna(0, inplace=True)

print("'time_diff_minutes' feature created successfully.")

def process_tags(df, max_len=10):
    # 填充缺失值为空字符串
    df['tag'] = df['tag'].fillna('')
    
    # 解析tag字符串为整数列表
    tag_lists = df['tag'].apply(lambda x: [int(i) for i in x.split(',')] if x != '' else [])
    
    # padding和truncating
    padded_tags = np.zeros((len(df), max_len), dtype=np.int64)
    for i, tags in enumerate(tag_lists):
        seq_len = min(len(tags), max_len)
        padded_tags[i, :seq_len] = tags[:seq_len]
        
    # 将处理好的padding数组转换为DataFrame的一列
    df['tag_processed'] = list(padded_tags)
    return df

print("Processing 'tag' feature...")
train_df = process_tags(train_df, max_len=10)
test_df = process_tags(test_df, max_len=10)
print(train_df['tag_processed'])
##############################################################

# ==================== 新增的TAG_VOCAB_SIZE计算代码 ====================
print("Calculating TAG_VOCAB_SIZE...")

# 1. 创建一个空集合，用于存放所有出现过的、不重复的tag ID
all_tags = set()

# 2. 遍历训练集中的tag列表，将所有tag ID加入集合
#    我们使用未padding的tag_lists来进行计算
train_tag_lists = train_df['tag'].apply(lambda x: [int(i) for i in x.split(',')] if x != '' else [])
for tags in train_tag_lists:
    all_tags.update(tags)

# 3. 遍历测试集中的tag列表，确保包含了所有可能的tag ID
test_tag_lists = test_df['tag'].apply(lambda x: [int(i) for i in x.split(',')] if x != '' else [])
for tags in test_tag_lists:
    all_tags.update(tags)

# 4. 计算词汇表大小
#    Embedding层的size需要是 (最大ID + 1)，因为ID是从0或1开始的
#    我们加上一个小的buffer（比如+1），以确保安全
if all_tags: # 确保集合不为空
    TAG_VOCAB_SIZE = max(all_tags) + 1
else:
    TAG_VOCAB_SIZE = 1 # 如果没有任何tag，至少为1

print(f"Total unique tags found: {len(all_tags)}")
print(f"Calculated TAG_VOCAB_SIZE: {TAG_VOCAB_SIZE}")
# ===================================================================



print(train_df.columns.tolist())

print("Step 3: Feature selection and engineering...")

# 定义稀疏和稠密特征

# sparse_features 列表无需更改，可直接使用
sparse_features = [
    # 原始ID类
    'user_id', 'video_id', 'author_id', 'music_id',

    # 交互特征
    'is_profile_enter', 
    
    # 用户特征
    'user_active_degree', 'is_live_streamer', 'is_video_author', 'is_lowactive_period',
    'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range',

    # 视频基础特征
    'video_type', 'upload_type', 'visible_status', 'music_type',

    # 场景特征
    'tab',

    # 加密的One-hot特征
] + [f'onehot_feat{i}' for i in range(18)]
#加密的物品

dense_features = [
    # 交互特征  这些交互的特征要去掉，因为会对数据进行泄露(和我们定义的因变量有一定的因果关系)
    # 'play_time_ms',
    # 'comment_stay_time',
    # 'profile_stay_time',
    'time_diff_minutes',

    # 用户原始数值特征 (已根据重命名方案更新)
    'user_register_days',       # 原名: register_days
    'user_following_count',     # 原名: follow_user_num
    'user_fan_count',           # 原名: fans_user_num
    'user_friend_count',        # 原名: friend_user_num
    
    # 视频基础数值特征 (已根据重命名方案更新)
    'video_total_duration',     # 原名: video_duration
    'server_width', 'server_height',
    
    # 视频统计特征 (所有相关特征已统一添加 'stat_' 前缀)
    'stat_show_cnt',              # 历史曝光数 (反映热度)
    'stat_play_cnt',              # 历史播放数 (反映热度)
    'stat_show_user_num',
    'stat_play_user_num',
    'stat_like_cnt',              # 历史点赞数 (其他任务的信号，可作为旁证)
    'stat_like_user_num',
    'stat_comment_cnt',           # 历史评论数 (同上)
    'stat_share_cnt',             # 历史分享数 (同上)
    'stat_share_user_num',
    'stat_follow_user_num',
    'stat_follow_cnt',            # 历史关注数 (同上)
    'stat_collect_cnt',           # 历史收藏数 (同上)
    'stat_short_time_play_cnt',    # 历史短播(跳出)数 (很好的负向信号)
]

'''对于一些需要预处理的特征, 可以在这里进行预处理之后添加到特征候选集中'''

# 填充缺失值
for col in sparse_features:
    train_df[col] = train_df[col].fillna('missing')
    test_df[col] = test_df[col].fillna('missing')

for col in dense_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0)



print("Step 4: Feature encoding...")
# --- 稀疏特征编码 ---
feature_max_idx = {}
# feature_max_idx['tagSize'] = TAG_VOCAB_SIZE

for col in sparse_features:
    le = LabelEncoder()

    # 在编码前，强制将该列统一转换为字符串类型，以解决混合类型问题 (混合类型报错, 每一个元素应该保持一样的类型)
    # 这样在转换之前, 就变成了字符串, 之后再统一转换成数字
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

    # 在完整数据上fit，保证编码一致性
    combined_series = pd.concat([train_df[col], test_df[col]], ignore_index=True)
    le.fit(combined_series)
    
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    
    # 记录每个特征的词汇表大小 (类别数量)
    feature_max_idx[col] = len(le.classes_)


file_path = os.path.join(PROCESSED_DATA_PATH, 'feature_max_idx_withoutTag.json')
# 保存特征词汇表信息
with open(file_path, 'w') as f:
    json.dump(feature_max_idx, f)

for col in dense_features:
    # 使用 pd.to_numeric，无法转换的值会变成NaN（无效数字）
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# 统一填充可能在转换过程中产生的NaN
train_df[dense_features] = train_df[dense_features].fillna(0)
test_df[dense_features] = test_df[dense_features].fillna(0)

# 强制转换为float32，这是PyTorch最常用的浮点类型
train_df[dense_features] = train_df[dense_features].astype('float32')
test_df[dense_features] = test_df[dense_features].astype('float32')


# --- 稠密特征归一化 ---
mms = StandardScaler()
# 在训练集上fit，然后对训练集和测试集transform
mms.fit(train_df[dense_features])

train_df[dense_features] = mms.transform(train_df[dense_features])
test_df[dense_features] = mms.transform(test_df[dense_features])

print("Step 5: Defining labels...")
# 定义CTR, CVR, CTCVR标签
train_df['ctr_label'] = train_df['is_click']
train_df['cvr_label'] = train_df['long_view']
train_df['ctcvr_label'] = train_df['is_click'] * train_df['long_view']

test_df['ctr_label'] = test_df['is_click']
test_df['cvr_label'] = test_df['long_view']
test_df['ctcvr_label'] = test_df['is_click'] * test_df['long_view']

print("Step 6: Saving processed data...")
# 选取需要的列并保存
label_cols = ['ctr_label', 'cvr_label', 'ctcvr_label']
# 标签特征
tags_col = ['tag_processed']
all_feature_cols = sparse_features + dense_features + tags_col

print(train_df[all_feature_cols + label_cols].dtypes.tolist())

train_file_path = os.path.join(PROCESSED_DATA_PATH, "train_data.pkl")
test_file_path = os.path.join(PROCESSED_DATA_PATH, "test_data.pkl")
online_test_file_path = os.path.join(PROCESSED_DATA_PATH, "online_test_data.pkl")
random_test_file_path = os.path.join(PROCESSED_DATA_PATH, "random_test_data.pkl")

online_test_df = test_df[test_df['is_rand'] == 0].reset_index(drop=True)
random_test_df = test_df[test_df['is_rand'] == 1].reset_index(drop=True)

train_df[all_feature_cols + label_cols].to_pickle(train_file_path)
test_df[all_feature_cols + label_cols].to_pickle(test_file_path)
online_test_df[all_feature_cols + label_cols].to_pickle(online_test_file_path)
random_test_df[all_feature_cols + label_cols].to_pickle(random_test_file_path)

print("Preprocessing finished successfully!")