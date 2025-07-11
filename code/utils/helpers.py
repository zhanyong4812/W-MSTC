# utils/helpers.py

from data.preprocess import load_data_cache
from config import BATCH_SIZE, TASK_NUM

indexes = {"train": 0, "test": 0}

def next_batch(mode, datasets, datasets_cache):
    """
    Get the next batch of data.
    :param mode: 'train' or 'test'
    :param datasets: dictionary of dataset identifiers
    :param datasets_cache: dictionary of cached data batches
    :return: the next batch of data
    """
    # If we've exhausted the cache for this mode, reset the index and reload
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    # Retrieve and return the next batch
    batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1
    return batch
