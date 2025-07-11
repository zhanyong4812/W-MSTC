import os
from datetime import datetime

# Base data directory
ROOT_DIR = '/data/dataset/MLRadio/RML2018.01a/Multi_SNR_6/'
# Default dataset file (for SNR=6)
TRAIN_DATA_TEMPLATE = os.path.join(ROOT_DIR, 'Mutil_SNR_6_combined_all.npy')

# Dataset split indices
TRAIN_SPLIT = 12
TEST_SPLIT = [12, 16, 18]

# Signal-to-noise ratio (to be set at runtime)
SNR = None

# Task parameters
N_WAY = 5           # Number of classes per episode
K_SPT = 5           # Number of support examples per class
K_QUERY = 5         # Number of query examples per class
TASK_NUM = 4        # Number of parallel tasks (meta-batch size)
BATCH_SIZE = TASK_NUM

# Image dimensions
RESIZE_WIDTH = 32
RESIZE_LENGTH = 32

# Stack sizes for input channels
STACK_SIZE = 10      # Total channels 
STACK_IMG_SIZE = 8   # Number of constellation image channels

# Cache settings
CACHE_BATCHES = 10   # Number of batches to keep in memory

# Model parameters
IMG_DIM = 256        # Dimension of image feature embedding
IQAP_DIM = 256       # Dimension of IQ amplitude-phase embedding

NUM_HEADS = 4        # Number of attention heads
EMBED_DIM = 256      # Embedding dimension for fusion

INPUT_DIM = 256
OUTPUT_DIM = 256
SEQ_LENGTH = 256     # Sequence length for transformer input
WINDOW_SIZES = [32, 64, 128]  # Attention window sizes per stage

# Swin-Transformer configuration
swin_params = {
    'num_layers': len(WINDOW_SIZES),                     # Number of transformer layers
    'window_sizes': WINDOW_SIZES,                        # Window size per layer
    'num_heads': [NUM_HEADS] * len(WINDOW_SIZES),        # Heads per layer
    'mlp_ratio': 4.0,                                    # MLP expansion ratio
    'drop': 0.2,                                         # Dropout rate
}

# Fusion branches toggle
branch = {
    "use_branch1": True,   # Windowed Attention
    "use_branch2": True,   # Multi-scale Self-Attention
    "use_branch3": True    # Cross-Attention with Swin-Transformer
}

# Number of transformer blocks per branch
block_layers = 1

# Training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 301

# Results directory for CSV output
DATA_DIR = './results/'

def get_dataset_path(snr):
    """
    Generate dataset file path for a given SNR.
    """
    return TRAIN_DATA_TEMPLATE.format(snr=snr)

def get_csv_filename(snr):
    """
    Generate a timestamped CSV filename for results, including N-way and K-shot.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_SNR{snr}_N{N_WAY}_K{K_SPT}_test.csv"
    return os.path.join(DATA_DIR, filename)
