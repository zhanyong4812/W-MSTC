# scripts/data_extractor.py
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import logging

MODULATION_LIST = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]
SNR_LIST = list(range(-20, 32, 2))  # -20, -18, ..., 30

def extract_iq_data_from_hdf5(hdf5_path, output_dir, target_modulations, target_snrs, samples_per_condition):
    """
    从给定 hdf5_path 中读取 X, Y, Z，然后根据目标调制、目标SNR、以及采样数提取 IQ 数据并保存到 output_dir。
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 文件不存在: {hdf5_path}")

    # 读取 HDF5 文件
    logging.info(f"打开HDF5文件: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        X = f['X'][:]  # (N, 1024, 2)
        Y = f['Y'][:]  # (N, 24)  (One-hot 编码)
        Z = f['Z'][:]  # (N, 1)
    logging.info(f"读取数据完成: X={X.shape}, Y={Y.shape}, Z={Z.shape}")

    # 遍历目标 SNR
    for snr in target_snrs:
        if snr not in SNR_LIST:
            logging.warning(f"SNR {snr} 不在预定义 SNR_LIST 中，跳过。")
            continue

        snr_folder = os.path.join(output_dir, f"SNR_{snr}")
        os.makedirs(snr_folder, exist_ok=True)

        # 遍历目标调制类型
        for mod_type in target_modulations:
            if mod_type not in MODULATION_LIST:
                logging.warning(f"调制类型 {mod_type} 不在预定义 MODULATION_LIST 中，跳过。")
                continue

            try:
                sel_X = _extract_single_mod_snr(X, Y, Z, mod_type, snr, samples_per_condition)
                file_name = f"X_SNR_{snr}_mod_{mod_type}.npy"
                np.save(os.path.join(snr_folder, file_name), sel_X)
                logging.info(f"保存 {file_name}, shape={sel_X.shape}")
            except ValueError as e:
                logging.error(e)

def _extract_single_mod_snr(X, Y, Z, mod_type, snr, num_samples):
    """
    内部函数：提取单一调制、单一SNR下的 num_samples 个 IQ 样本。
    """
    mod_index = MODULATION_LIST.index(mod_type)
    snr_mask = (Z[:, 0] == snr)

    # Y[:, mod_index] == 1 表示此条目属于该调制类型
    mod_mask = (Y[:, mod_index] == 1)
    combined_mask = mod_mask & snr_mask
    selected_indices = np.where(combined_mask)[0]

    total_available = len(selected_indices)
    if total_available < num_samples:
        raise ValueError(f"{mod_type}, SNR={snr}: 可用({total_available}) < 请求({num_samples})")

    random_indices = np.random.choice(selected_indices, size=num_samples, replace=False)
    return X[random_indices]  # shape: (num_samples, 1024, 2)
