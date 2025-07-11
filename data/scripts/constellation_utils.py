# scripts/constellation_utils.py
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os

def map_to_grid(values, grid_size=64):
    """
    Map values in the range [-1, 1] to integer indices in [0, grid_size-1].
    """
    # Normalize from [-1,1] to [0,1]
    normalized = (values + 1) / 2  
    # Scale to [0, grid_size-1] and convert to int
    indices = (normalized * (grid_size - 1)).astype(int)
    return np.clip(indices, 0, grid_size - 1)

def generate_constellation_data(iq_data, output_file, grid_size=64, num_timesteps=4):
    """
    Generate example constellation diagrams from IQ data.
    """
    logging.info(f"Starting constellation generation, input shape={iq_data.shape}")
    c, s, _, _, length = iq_data.shape
    points_per_timestep = length // num_timesteps 

    constellation = np.zeros((c, s, num_timesteps, 1, grid_size, grid_size),
                             dtype=np.int32)

    for mod in range(c):
        for sample_idx in range(s):
            # Extract the 2×length IQ sample
            sample = iq_data[mod, sample_idx, 0, :, :]
            I = sample[0].astype(np.float32)
            Q = sample[1].astype(np.float32)

            # Normalize this sample to [-1,1]
            max_abs = max(np.max(np.abs(I)), np.max(np.abs(Q)), 1e-8)
            I /= max_abs
            Q /= max_abs

            for t in range(num_timesteps):
                start = t * points_per_timestep
                end = start + points_per_timestep
                I_segment = I[start:end]
                Q_segment = Q[start:end]

                i_indices = map_to_grid(I_segment, grid_size)
                q_indices = map_to_grid(Q_segment, grid_size)
                # Accumulate into the constellation histogram
                np.add.at(constellation[mod, sample_idx, t, 0],
                          (i_indices, q_indices), 1)

    np.save(output_file, constellation)
    logging.info(f"Constellation data saved to {output_file}, shape={constellation.shape}")
