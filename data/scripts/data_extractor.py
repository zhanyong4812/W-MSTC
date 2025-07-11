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
    Read datasets X, Y, Z from the specified HDF5 file, then extract IQ samples
    according to the target modulations, target SNRs, and number of samples,
    and save the results into output_dir.
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    # Open the HDF5 file
    logging.info(f"Opening HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        X = f['X'][:]  # shape: (N, 1024, 2)
        Y = f['Y'][:]  # shape: (N, 24) one-hot encoding of modulations
        Z = f['Z'][:]  # shape: (N, 1) SNR values
    logging.info(f"Data loaded: X={X.shape}, Y={Y.shape}, Z={Z.shape}")

    # Iterate over each desired SNR
    for snr in target_snrs:
        if snr not in SNR_LIST:
            logging.warning(f"SNR {snr} is not in predefined SNR_LIST, skipping.")
            continue

        snr_folder = os.path.join(output_dir, f"SNR_{snr}")
        os.makedirs(snr_folder, exist_ok=True)

        # Iterate over each desired modulation type
        for mod_type in target_modulations:
            if mod_type not in MODULATION_LIST:
                logging.warning(f"Modulation {mod_type} is not in predefined MODULATION_LIST, skipping.")
                continue

            try:
                sel_X = _extract_single_mod_snr(X, Y, Z, mod_type, snr, samples_per_condition)
                file_name = f"X_SNR_{snr}_mod_{mod_type}.npy"
                np.save(os.path.join(snr_folder, file_name), sel_X)
                logging.info(f"Saved {file_name}, shape={sel_X.shape}")
            except ValueError as e:
                logging.error(e)

def _extract_single_mod_snr(X, Y, Z, mod_type, snr, num_samples):
    """
    Internal helper: extract `num_samples` IQ samples for a single modulation type
    and a single SNR from arrays X, Y, Z.
    """
    mod_index = MODULATION_LIST.index(mod_type)
    snr_mask = (Z[:, 0] == snr)

    # Y[:, mod_index] == 1 indicates this sample has the target modulation
    mod_mask = (Y[:, mod_index] == 1)
    combined_mask = mod_mask & snr_mask
    selected_indices = np.where(combined_mask)[0]

    total_available = len(selected_indices)
    if total_available < num_samples:
        raise ValueError(f"{mod_type}, SNR={snr}: available({total_available}) < requested({num_samples})")

    random_indices = np.random.choice(selected_indices, size=num_samples, replace=False)
    return X[random_indices]  # shape: (num_samples, 1024, 2)
