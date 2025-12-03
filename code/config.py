# config.py
"""
Configuration module that loads settings from config.yaml and exposes them as module attributes.
This module maintains backward compatibility with existing code that imports from config.
"""

import os
import yaml
from pathlib import Path

# Load configuration from YAML file (avoid circular import by implementing locally)
def _load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

_config = _load_config()

# ============================================================================
# Data Configuration
# ============================================================================
ROOT_DIR = _config['data']['root_dir']
TRAIN_DATA_TEMPLATE = os.path.join(ROOT_DIR, _config['data']['train_data_template'])
TRAIN_SPLIT = _config['data']['train_split']
TEST_SPLIT = _config['data']['test_split']

# Runtime data paths (will be set at runtime)
TRAIN_DATA = None
SNR = None

# ============================================================================
# Task Parameters
# ============================================================================
N_WAY = _config['task']['n_way']
K_SPT = _config['task']['k_spt']
K_QUERY = _config['task']['k_query']
TASK_NUM = _config['task']['task_num']
BATCH_SIZE = _config['task']['batch_size']

# ============================================================================
# Image Dimensions
# ============================================================================
RESIZE_WIDTH = _config['image']['resize_width']
RESIZE_LENGTH = _config['image']['resize_length']

# ============================================================================
# Channel Configuration
# ============================================================================
STACK_SIZE = _config['channels']['stack_size']
STACK_IMG_SIZE = _config['channels']['stack_img_size']

# ============================================================================
# Cache Settings
# ============================================================================
CACHE_BATCHES = _config['cache']['cache_batches']

# ============================================================================
# Model Parameters
# ============================================================================
IMG_DIM = _config['model']['img_dim']
IQAP_DIM = _config['model']['iqap_dim']
NUM_HEADS = _config['model']['num_heads']
EMBED_DIM = _config['model']['embed_dim']
INPUT_DIM = _config['model']['input_dim']
OUTPUT_DIM = _config['model']['output_dim']
SEQ_LENGTH = _config['model']['seq_length']
WINDOW_SIZES = _config['model']['window_sizes']

# ============================================================================
# Swin-Transformer Configuration
# ============================================================================
swin_params = {
    'num_layers': _config['swin_transformer']['num_layers'],
    'window_sizes': WINDOW_SIZES,
    'num_heads': [NUM_HEADS] * len(WINDOW_SIZES),
    'mlp_ratio': _config['swin_transformer']['mlp_ratio'],
    'drop': _config['swin_transformer']['drop'],
}

# ============================================================================
# Fusion Branches
# ============================================================================
branch = {
    "use_wa": _config['branches']['use_wa'],
    "use_mwa": _config['branches']['use_mwa'],
    "use_ca": _config['branches']['use_ca'],
}

# ============================================================================
# Transformer Configuration
# ============================================================================
block_layers = _config['transformer']['block_layers']

# ============================================================================
# Training Hyperparameters
# ============================================================================
LEARNING_RATE = _config['training']['learning_rate']
NUM_EPOCHS = _config['training']['num_epochs']

# ============================================================================
# Results Directory
# ============================================================================
DATA_DIR = _config['results']['data_dir']

# ============================================================================
# Supervised Classification (non-few-shot) Settings
# ============================================================================
CLS_NUM_CLASSES = _config['classification']['num_classes']
CLS_USE_CLASSES = _config['classification']['use_classes']
CLS_BATCH_SIZE = _config['classification']['batch_size']
CLS_NUM_WORKERS = _config['classification']['num_workers']
CLS_SAVE_DIR = _config['classification']['save_dir']

# ============================================================================
# Backward Compatibility Functions
# These functions are kept for backward compatibility but delegate to utils
# ============================================================================
def get_dataset_path(snr=None):
    """
    Generate dataset file path for a given SNR.
    Template can contain {snr}, {samples}, {size}, {step}.
    """
    root_dir = _config['data']['root_dir']
    template = _config['data']['train_data_template']

    fmt_kwargs = {}
    if '{snr}' in template and snr is not None:
        fmt_kwargs['snr'] = snr
    # Optional parameters: samples / size / step
    for key in ('samples', 'size', 'step'):
        if '{' + key + '}' in template and key in _config['data']:
            fmt_kwargs[key] = _config['data'][key]

    if fmt_kwargs:
        template = template.format(**fmt_kwargs)

    return os.path.join(root_dir, template)


def get_csv_filename(snr, n_way=None, k_spt=None):
    """
    Generate a timestamped CSV filename for results.
    Maintains backward compatibility with existing code.
    """
    from datetime import datetime
    
    data_dir = _config['results']['data_dir']
    
    # Create SNR-specific subdirectory
    snr_dir = os.path.join(data_dir, f"SNR_{snr}")
    os.makedirs(snr_dir, exist_ok=True)
    
    # Get task parameters
    n_way = n_way or _config['task']['n_way']
    k_spt = k_spt or _config['task']['k_spt']
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_SNR{snr}_N{n_way}_K{k_spt}_test.csv"
    
    return os.path.join(snr_dir, filename)
