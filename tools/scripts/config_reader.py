# scripts/config_reader.py
# -*- coding: utf-8 -*-

import os
import yaml

def read_config(config_filename="config.yaml"):
    """
    Read a YAML configuration file from the config/ directory and return it as a dictionary.
    """
    # Assume this file is located in scripts/, and the config directory is ../config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg
