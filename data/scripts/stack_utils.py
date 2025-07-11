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
    Collect files named 'X_SNR_<snr>_mod_<modulation>.npy' in the given folder,
    and stack them in the order of MODULATION_ORDER. Returns an array of shape
    (24, N, 1024, 2) and the SNR value.
    """
    file_list = os.listdir(folder_path)
    snr_pattern = re.compile(r"X_SNR_([+-]?\d+)_mod_(.*)\.npy")

    data_list = []
    snr_value = None

    for mod in tqdm(MODULATION_ORDER, desc="Stacking IQ data"):
        # Find the first matching file for this modulation
        matching_files = [f for f in file_list if f"mod_{mod}" in f and snr_pattern.search(f)]
        if matching_files:
            file_path = os.path.join(folder_path, matching_files[0])
            # Extract the SNR from the filename
            match = snr_pattern.search(file_path)
            if match:
                snr_value = match.group(1)
            data = np.load(file_path)      # expected shape: (N, 1024, 2)
            # Transpose to (N, 2, 1024)
            data = data.transpose(0, 2, 1)
            # Add a singleton dimension for consistency: (N, 1, 2, 1024)
            data = np.expand_dims(data, axis=1)
            data_list.append(data)
        else:
            logging.warning(f"No file for modulation {mod} found in {folder_path}.")

    if not data_list:
        raise ValueError(f"No matching IQ files found in {folder_path}!")

    # Stack across the modulation axis: (24, N, 1, 2, 1024)
    combined_data = np.stack(data_list, axis=0)
    logging.info(f"Stacking complete: combined_data.shape={combined_data.shape}, SNR={snr_value}")
    return combined_data, snr_value

def get_reshaped_data(m1, m2, grid_size, num_timesteps):
    """
    Reshape the constellation data (m1) and IQ data (m2) according to
    the specified grid_size and num_timesteps. Returns (m1_reshaped, m2_reshaped).
    """
    # NOTE: Update these branches to match your actual data dimensions.
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
    Load constellation (.npy) and IQ (.npy) files, reshape them using get_reshaped_data,
    concatenate along the channel axis, and save to output_file.
    """
    m1 = np.load(constellation_file)
    m2 = np.load(iq_file)
    logging.info(f"Loaded constellation: {m1.shape}, IQ: {m2.shape}")

    # Reshape both arrays
    m1_reshaped, m2_reshaped = get_reshaped_data(m1, m2, grid_size, num_timesteps)

    # Concatenate along axis 2 (channel dimension)
    combined = np.concatenate([m1_reshaped, m2_reshaped], axis=2)

    np.save(output_file, combined)
    logging.info(f"Stacking complete: {combined.shape}, saved to {output_file}")
