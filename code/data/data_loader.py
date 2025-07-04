import numpy as np
from sklearn.model_selection import train_test_split
import config

def load_data(test_size=0.2, random_state=42):
    """
    读取原始堆栈数据并按样本层面做 8:2 划分：
      - config.TRAIN_DATA 指向的 .npy 文件形状假设为 (C, M, STACK, 1, 32, 32)。
      - C = 类别数，M = 每类样本数，STACK = 每个样本的帧数（例如 10 = IMG_STACK + IQ_STACK）。
      如果在 config.USE_CLASSES 中指定了 N（<=C），则只使用前 N 类进行实验。
      返回字典：
        {
          'train': {'img': X_train, 'label': y_train},
          'test':  {'img': X_test,  'label': y_test}
        }
      其中 X_train.shape = (N_train, STACK, 1, 32, 32)，y_train.shape = (N_train,)。
    """
    # 1. 读取所有数据（C 类、每类 M 条、每条 STACK 帧、1×32×32）
    img_list = np.load(config.TRAIN_DATA)  # e.g. shape = (11, 1000, 10, 1, 32, 32)
    C, M = img_list.shape[0], img_list.shape[1]

    # 如果在 config.USE_CLASSES 中指定要使用前 N 类，就先截取
    # 假设 config.USE_CLASSES = 5，就会把原本 (11,1000, …) 截成 (5,1000,…)
    if hasattr(config, 'USE_CLASSES') and config.USE_CLASSES is not None:
        N = config.USE_CLASSES
        assert 1 <= N <= C, f"USE_CLASSES 必须在 1 到 {C} 之间"
        img_list = img_list[:N]   # 只保留前 N 个类别
        C = N

    # 2. 扁平化： (C, M, STACK, 1, 32, 32) -> (C*M, STACK, 1, 32, 32)
    X = img_list.reshape(C * M, *img_list.shape[2:])

    # 3. 生成标签：每 M 条属于同一类（此时新 C = N）
    y = np.zeros(C * M, dtype=np.int64)
    for class_idx in range(C):
        start = class_idx * M
        end = (class_idx + 1) * M
        y[start:end] = class_idx

    # 4. 随机打乱并按 8:2 划分，保持各类比例不变
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"DB: train {X_train.shape}, test {X_test.shape}  （使用前 {C} 类）")
    return {
        'train': {'img': X_train, 'label': y_train},
        'test':  {'img': X_test,  'label': y_test}
    }
