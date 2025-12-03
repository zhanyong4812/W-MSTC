# data/data_loader.py

import numpy as np
import os
from config import TRAIN_SPLIT, TEST_SPLIT, ROOT_DIR
import config

def load_data():
    # Expected shape: (24, N, 10, 1, 32, 32) where
    # 24 = number of modulation types
    # N = number of samples per modulation (e.g., 1000)
    # 10 = total channels (8 constellation + 2 IQ)
    # 1 = singleton dimension
    # 32x32 = spatial dimensions
    img_list = np.load(config.TRAIN_DATA)
    x_train = img_list[:TRAIN_SPLIT]
    x_test = np.concatenate([img_list[TEST_SPLIT[0]:TEST_SPLIT[1]], img_list[TEST_SPLIT[2]:]], axis=0)
    num_classes = img_list.shape[0]
    datasets = {'train': x_train, 'test': x_test}
    print("DB: train", x_train.shape, "test", x_test.shape)
    return datasets
