# data/preprocess.py
import numpy as np
import config

def load_data_cache(dataset):
    """
    Collects several batches data for N-shot learning.
    :param dataset: [cls_num, samples, channels, height, width]
    :return: A list with [support_set_x, support_set_y, target_x, target_y]
    """
    setsz = config.K_SPT * config.N_WAY
    querysz = config.K_QUERY * config.N_WAY
    data_cache = []

    for _ in range(config.CACHE_BATCHES):
        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for _ in range(config.BATCH_SIZE):
            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], config.N_WAY, replace=False)
            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(dataset.shape[1], config.K_SPT + config.K_QUERY, replace=False)
                x_spt.append(dataset[cur_class][selected_img[:config.K_SPT]])
                x_qry.append(dataset[cur_class][selected_img[config.K_SPT:]])
                y_spt.append([j] * config.K_SPT)
                y_qry.append([j] * config.K_QUERY)
            perm = np.random.permutation(config.N_WAY * config.K_SPT)
            x_spt = np.array(x_spt).reshape(config.N_WAY * config.K_SPT, config.STACK_SIZE, 1, config.RESIZE_WIDTH, config.RESIZE_LENGTH)[perm]
            y_spt = np.array(y_spt).reshape(config.N_WAY * config.K_SPT)[perm]
            perm = np.random.permutation(config.N_WAY * config.K_QUERY)
            x_qry = np.array(x_qry).reshape(config.N_WAY * config.K_QUERY, config.STACK_SIZE, 1, config.RESIZE_WIDTH, config.RESIZE_LENGTH)[perm]
            y_qry = np.array(y_qry).reshape(config.N_WAY * config.K_QUERY)[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_spts = np.array(x_spts).astype(np.float32).reshape(config.BATCH_SIZE, setsz, config.STACK_SIZE, 1, config.RESIZE_WIDTH, config.RESIZE_LENGTH)
        y_spts = np.array(y_spts).astype(np.int64).reshape(config.BATCH_SIZE, setsz)
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(config.BATCH_SIZE, querysz, config.STACK_SIZE, 1, config.RESIZE_WIDTH, config.RESIZE_LENGTH)
        y_qrys = np.array(y_qrys).astype(np.int64).reshape(config.BATCH_SIZE, querysz)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache
