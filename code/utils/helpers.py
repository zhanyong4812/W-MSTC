# utils/helpers.py

from data.preprocess import load_data_cache
from config import BATCH_SIZE, TASK_NUM

indexes = {"train": 0, "test": 0}

def next_batch(mode, datasets, datasets_cache):
    """
    获取下一个批次的数据
    :param mode: 'train' 或 'test'
    :param datasets: 数据集字典
    :param datasets_cache: 缓存的数据字典
    :return: 下一个批次的数据
    """
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1
    return next_batch
