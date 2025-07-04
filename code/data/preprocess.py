import numpy as np
import config

def load_data_cache(dataset):
    """
    小样本(N-shot)逻辑已移除。
    当前流程为常规 8:2 划分，不再使用“支持集/查询集”的概念，
    如果误调用此函数，将抛出异常提醒。
    """
    raise NotImplementedError(
        "load_data_cache 已移除。当前流程改为常规 8:2 划分，不再使用支持集/查询集。"
    )
