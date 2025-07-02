# scripts/stack_utils.py
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from tqdm import tqdm
import logging

MODULATION_ORDER = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

def stack_iq_data_from_folder(folder_path):
    """
    在给定的文件夹中，搜集形如 'X_SNR_xx_mod_xxx.npy' 的文件，
    按照 MODULATION_ORDER 顺序堆叠为 (24, N, 10000024, 2) 并返回。
    """
    file_list = os.listdir(folder_path)
    snr_pattern = re.compile(r"X_SNR_([+-]?\d+)_mod_(.*)\.npy")


    data_list = []
    snr_value = None

    for mod in tqdm(MODULATION_ORDER, desc="Stacking IQ data"):
        matching_files = [f for f in file_list if f"mod_{mod}" in f and snr_pattern.search(f)]
        if matching_files:
            file_path = os.path.join(folder_path, matching_files[0])
            # 提取 snr
            match = snr_pattern.search(file_path)
            if match:
                snr_value = match.group(1)
            data = np.load(file_path)  # (N, 10000024, 2)
            # 1) 转置: (N, 10000024, 2) -> (N, 2, 10000024)
            data = data.transpose(0, 2, 1)
            data = np.expand_dims(data, axis=1)  # (N, 1, 2, 10000024)
            data_list.append(data)
        else:
            logging.warning(f"在 {folder_path} 中未找到 {mod} 对应文件.")

    if not data_list:
        raise ValueError(f"未在 {folder_path} 中找到任何匹配文件！")

    combined_data = np.stack(data_list, axis=0)  # (24, N, 10000024, 2)
    logging.info(f"堆叠完成: combined_data.shape={combined_data.shape}, SNR={snr_value}")
    return combined_data, snr_value

def get_reshaped_data(m1, m2, grid_size, num_timesteps):
    """
    根据不同的 grid_size 和 num_timesteps 调整数据形状
    根据你提供的组合要求来调整每个分支。
    """
    if grid_size == 16:
        if num_timesteps == 1:
            return m1.reshape(24, 1000, 1, 1, 16, 16), m2.reshape(24, 1000, 8, 1, 16, 16)
        elif num_timesteps == 2:
            return m1.reshape(24, 1000, 2, 1, 16, 16), m2.reshape(24, 1000, 8, 1, 16, 16)
        elif num_timesteps == 4:
            return m1.reshape(24, 1000, 4, 1, 16, 16), m2.reshape(24, 1000, 8, 1, 16, 16)
        elif num_timesteps == 8:
            return m1.reshape(24, 1000, 8, 1, 16, 16), m2.reshape(24, 1000, 8, 1, 16, 16)
        elif num_timesteps == 16:
            return m1.reshape(24, 1000, 16, 1, 16, 16), m2.reshape(24, 1000, 8, 1, 16, 16)

    elif grid_size == 32:
        if num_timesteps == 1:
            return m1.reshape(24, 1000, 1, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 2:
            return m1.reshape(24, 1000, 2, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 4:
            return m1.reshape(24, 1000, 4, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 8:
            return m1.reshape(24, 1000, 8, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 16:
            return m1.reshape(24, 1000, 16, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)

    elif grid_size == 64:
        if num_timesteps == 1:
            return m1.reshape(24, 1000, 4, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 2:
            return m1.reshape(24, 1000, 8, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 4:
            return m1.reshape(24, 1000, 16, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 8:
            return m1.reshape(24, 1000, 32, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 16:
            return m1.reshape(24, 1000, 64, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)

    elif grid_size == 128:
        if num_timesteps == 1:
            return m1.reshape(24, 1000, 16, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 2:
            return m1.reshape(24, 1000, 32, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 4:
            return m1.reshape(24, 1000, 64, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 8:
            return m1.reshape(24, 1000, 128, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
        elif num_timesteps == 16:
            return m1.reshape(24, 1000, 512, 1, 32, 32), m2.reshape(24, 1000, 2, 1, 32, 32)
    else:
        raise ValueError(f"Unsupported grid_size: {grid_size}")


def stack_constellation_and_iq(constellation_file, iq_file, output_file, grid_size, num_timesteps):
    """
    将星座图和 IQ 数据拼接到一起并保存，调整形状以适应不同的 grid_size 和 num_timesteps。
    """
    m1 = np.load(constellation_file)
    m2 = np.load(iq_file)
    logging.info(f"加载星座图: {m1.shape}, IQ图: {m2.shape}")

    # 调整 m1 和 m2 的形状
    m1_reshaped, m2_reshaped = get_reshaped_data(m1,m2, grid_size, num_timesteps)

    # 在轴2（通道轴）上拼接
    combined = np.concatenate([m1_reshaped, m2_reshaped], axis=2)

    np.save(output_file, combined)
    logging.info(f"拼接完成: {combined.shape}, 保存至 {output_file}")
