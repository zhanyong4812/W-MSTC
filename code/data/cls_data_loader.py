import numpy as np
from sklearn.model_selection import train_test_split
import config


def load_flat_data(test_size=0.2, random_state=42):
    """
    Data loader for non-few-shot supervised classification:
      - .npy file pointed by config.TRAIN_DATA is assumed to have shape (C, M, STACK, 1, 32, 32)
        C = number of classes, M = samples per class, STACK = channel count (e.g. 10 = 8 IMG + 2 IQ)
      - Can use only first N classes via config.CLS_USE_CLASSES

    Returns:
      {
        'train': {'img': X_train, 'label': y_train},
        'test':  {'img': X_test,  'label': y_test}
      }
    where X_train.shape = (N_train, STACK, 1, 32, 32)
    """
    img_list = np.load(config.TRAIN_DATA)  # (C, M, STACK, 1, 32, 32)
    C, M = img_list.shape[0], img_list.shape[1]

    # Use only first N classes (optional)
    if config.CLS_USE_CLASSES is not None:
        N = config.CLS_USE_CLASSES
        assert 1 <= N <= C, f"CLS_USE_CLASSES must be between 1 and {C}"
        img_list = img_list[:N]
        C = N

    # Flatten to sample dimension
    X = img_list.reshape(C * M, *img_list.shape[2:])  # (C*M, STACK, 1, 32, 32)

    # Generate labels: every M samples belong to the same class
    y = np.zeros(C * M, dtype=np.int64)
    for class_idx in range(C):
        start = class_idx * M
        end = (class_idx + 1) * M
        y[start:end] = class_idx

    # Stratified random split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[CLS] DB: train {X_train.shape}, test {X_test.shape}  (using first {C} classes)")
    return {
        "train": {"img": X_train, "label": y_train},
        "test": {"img": X_test, "label": y_test},
    }


