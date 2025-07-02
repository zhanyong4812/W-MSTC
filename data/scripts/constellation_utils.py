# scripts/constellation_utils.py
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os

def map_to_grid(values, grid_size=64):
    """
    将取值范围在 [-1,1] 的 values 映射到 [0, grid_size-1] 的整数索引上。
    """
    normalized = (values + 1) / 2  # [-1,1] -> [0,1]
    indices = (normalized * (grid_size - 1)).astype(int)
    return np.clip(indices, 0, grid_size - 1)

def generate_constellation_data(iq_data, output_file, grid_size=64, num_timesteps=4):
    """
    根据 IQ 数据生成星座图的示例。
    假设 iq_data.shape = (24, 1000, 1, 2, 1024) 或类似形状，需要根据实际调整。
    最终生成星座图形状 (24, 1000, num_timesteps, 1, grid_size, grid_size)。
    """
    logging.info(f"开始生成星座图, shape={iq_data.shape}")
    c, s, _, _, length = iq_data.shape
    points_per_timestep = length // num_timesteps  # 假设能整除

    constellation = np.zeros((c, s, num_timesteps, 1, grid_size, grid_size), dtype=np.int32)

    for mod in range(c):
        for sample_idx in range(s):
            # (2, length)
            sample = iq_data[mod, sample_idx, 0, :, :]
            I = sample[0]
            Q = sample[1]

            for t in range(num_timesteps):
                start = t * points_per_timestep
                end = start + points_per_timestep
                I_segment = I[start:end]
                Q_segment = Q[start:end]

                i_indices = map_to_grid(I_segment, grid_size)
                q_indices = map_to_grid(Q_segment, grid_size)
                # 累加星座图
                np.add.at(constellation[mod, sample_idx, t, 0], (i_indices, q_indices), 1)

    np.save(output_file, constellation)
    logging.info(f"星座图已保存: {output_file}, shape={constellation.shape}")
