# config.py
import os

# --- 数据路径 ---
DATA_PATH = "../data/"
LOG_STANDARD_TRAIN = DATA_PATH + "log_standard_4_08_to_4_21_pure.csv"
LOG_STANDARD_TEST = DATA_PATH + "log_standard_4_22_to_5_08_pure.csv"
LOG_RANDOM_TEST = DATA_PATH + "log_random_4_22_to_5_08_pure.csv"
USER_FEATURES = DATA_PATH + "user_features_pure.csv"
VIDEO_BASIC_FEATURES = DATA_PATH + "video_features_basic_pure.csv"
VIDEO_STAT_FEATURES = DATA_PATH + "video_features_statistic_pure.csv"


FEATURE_INFO_PATH = os.path.join(DATA_PATH, 'processed/feature_max_idx_withoutTag.json')
ONLINE_TEST_PKL_PATH = DATA_PATH + 'processed/online_test_data.pkl'
RANDOM_TEST_PKL_PATH = DATA_PATH + 'processed/random_test_data.pkl'

TRAIN_PKL_PATH = DATA_PATH + 'processed/train_data.pkl'
TEST_PKL_PATH = DATA_PATH + 'processed/test_data.pkl'


# --- 输出路径 ---
PROCESSED_DATA_PATH = "./processed/"
SAVED_MODEL_PATH = "./checkpoints/"
LOG_TRAIN_PATH = "./logs/"
ROOT_PATH = './'

# --- 模型超参数 ---
EMBEDDING_DIM = 24
NUM_EXPERTS = 4
HIDDEN_UNITS = {'expert':[512, 128], 'tower':[16]}

# --- 训练超参数 ---
LEARNING_RATE = 0.003
BATCH_SIZE = 128
EPOCHS = 2
DEVICE = "cuda" # 如果有GPU，使用 "cuda"；否则使用 "cpu"

# tags max len
MAX_TAGS_LEN = 4
NUM_TASKS = 2
# tags的类别数:
COUNT_TAGS = 69

TASKS = ['ctr', 'cvr']

EXPERT_OUTPUT_DIM = 32

# SPARSE_FEATURES = ['user_id', 'video_id', 'author_id', 'music_id', 'is_profile_enter', 
#                    'user_active_degree', 'is_live_streamer', 'is_video_author', 
#                    'is_lowactive_period', 'follow_user_num_range', 'fans_user_num_range', 
#                    'friend_user_num_range', 'register_days_range', 'video_type', 'upload_type', 
#                    'visible_status', 'music_type', 'tab', 'onehot_feat0', 'onehot_feat1', 
#                    'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 
#                    'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 
#                    'onehot_feat10', 'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 
#                    'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17']

SPARSE_FEATURES = ['user_id', 'video_id', 'author_id', 'music_id', 'is_profile_enter', 
                   'user_active_degree', 'is_live_streamer', 'is_video_author', 
                   'is_lowactive_period', 'follow_user_num_range', 'fans_user_num_range', 
                   'friend_user_num_range', 'register_days_range', 'video_type', 'upload_type', 
                   'visible_status', 'music_type', 'tab', 'onehot_feat0', 'onehot_feat1', 
                   'onehot_feat2', 'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 
                   'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 
                   'onehot_feat10', 'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 
                   'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17']

DENSE_FEATURES = ['time_diff_minutes', 'user_register_days', 'user_following_count', 
                  'user_fan_count', 'user_friend_count', 'video_total_duration', 
                  'server_width', 'server_height', 
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
                    'stat_short_time_play_cnt',
                  ]
