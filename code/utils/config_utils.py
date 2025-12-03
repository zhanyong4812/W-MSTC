# utils/config_utils.py
"""
Configuration utility functions for loading and managing configuration.
"""

import os
import yaml
from datetime import datetime
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, searches for config.yaml in code directory.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / 'config.yaml'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_dataset_path(config, snr=None):
    """
    Generate dataset file path for a given SNR.
    
    Args:
        config: Configuration dictionary
        snr: Signal-to-noise ratio (optional, for future use)
    
    Returns:
        str: Full path to dataset file
    """
    root_dir = config['data']['root_dir']
    template = config['data']['train_data_template']
    
    # If template contains {snr}, format it
    if '{snr}' in template and snr is not None:
        template = template.format(snr=snr)
    
    return os.path.join(root_dir, template)


def get_csv_filename(config, snr, n_way=None, k_spt=None):
    """
    Generate a timestamped CSV filename for results.
    
    Args:
        config: Configuration dictionary
        snr: Signal-to-noise ratio
        n_way: Number of classes per episode (optional, uses config if not provided)
        k_spt: Number of support examples per class (optional, uses config if not provided)
    
    Returns:
        str: Full path to CSV file
    """
    data_dir = config['results']['data_dir']
    
    # Create SNR-specific subdirectory
    snr_dir = os.path.join(data_dir, f"SNR_{snr}")
    os.makedirs(snr_dir, exist_ok=True)
    
    # Get task parameters
    n_way = n_way or config['task']['n_way']
    k_spt = k_spt or config['task']['k_spt']
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_SNR{snr}_N{n_way}_K{k_spt}_test.csv"
    
    return os.path.join(snr_dir, filename)


def update_runtime_config(config, snr=None, **kwargs):
    """
    Update runtime configuration parameters.
    
    Args:
        config: Configuration dictionary (will be modified in place)
        snr: Signal-to-noise ratio
        **kwargs: Additional runtime parameters to update
    
    Returns:
        dict: Updated configuration dictionary
    """
    if 'runtime' not in config:
        config['runtime'] = {}
    
    if snr is not None:
        config['runtime']['snr'] = snr
        config['runtime']['train_data'] = get_dataset_path(config, snr)
        # Update data_dir to include SNR subdirectory
        config['results']['data_dir'] = os.path.join(
            config['results']['data_dir'], 
            f"SNR_{snr}"
        )
    
    # Update any other runtime parameters
    for key, value in kwargs.items():
        config['runtime'][key] = value
    
    return config

