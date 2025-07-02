import os
from datetime import datetime

# 数据路径
ROOT_DIR = '/data/dataset/MLRadio/RML2018.01a/Multi_SNR_6/'
# Default dataset (for SNR=6)
TRAIN_DATA_TEMPLATE = os.path.join(ROOT_DIR, 'Mutil_SNR_6_combined_all.npy')

# 数据集分割
TRAIN_SPLIT = 12
TEST_SPLIT = [12, 16, 18]

SNR = None
# 任务参数
N_WAY = 5
K_SPT = 5
K_QUERY = 5
TASK_NUM = 4
BATCH_SIZE = TASK_NUM

# 图像尺寸
RESIZE_WIDTH = 32
RESIZE_LENGTH = 32

STACK_SIZE = 10  # 表示 16+2，代表星座图和 IQ 图像
STACK_IMG_SIZE = 8
# 缓存参数
CACHE_BATCHES = 10

# 模型参数
IMG_DIM = 256
IQAP_DIM = 256

NUM_HEADS = 4
EMBED_DIM = 256

INPUT_DIM = 256
OUTPUT_DIM = 256
SEQ_LENGTH = 256  # Total of two seq length
WINDOW_SIZES = [32, 64, 128]

# swin-transformer 参数
swin_params = {
    'num_layers': len(WINDOW_SIZES),        # 每个窗口大小对应一个层
    'window_sizes': WINDOW_SIZES,             # 每个阶段的窗口大小
    'num_heads': [NUM_HEADS] * len(WINDOW_SIZES),  # 每个阶段的头数
    'mlp_ratio': 4.0,
    'drop': 0.2,
}

branch = {
    "use_branch1": True,  # WA
    "use_branch2": True,  # MWA
    "use_branch3": True   # CA
}

block_layers = 1

# 训练参数
LEARNING_RATE = 0.001
NUM_EPOCHS = 301

# CSV 保存路径相关参数
DATA_DIR = './results/'

# 动态生成SNR路径和输出文件
def get_dataset_path(snr):
    return TRAIN_DATA_TEMPLATE.format(snr=snr)

def get_csv_filename(snr):
    _SCRIPT_NAME = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_SNR{snr}"
    return os.path.join(DATA_DIR, f"{_SCRIPT_NAME}_N{N_WAY}_K{K_SPT}_test.csv")
