# scripts/config_reader.py
# -*- coding: utf-8 -*-

import os
import yaml

def read_config(config_filename="config.yaml"):
    """
    从 config/ 目录下读取 YAML 配置文件并返回字典。
    """
    # 假设本文件位于 scripts 文件夹，config 位于上级目录 ../config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg
