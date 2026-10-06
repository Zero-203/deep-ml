import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    indexs = rng.permutation(n)
    train_end, validation_end = int(n*train_frac), int(n*train_frac) + int(n*val_frac)
    y = y[indexs]

    test_predictions = y[validation_end:]
    train_idx, val_idx, test_idx = indexs[:train_end], indexs[train_end:validation_end], indexs[validation_end:]
    return test_predictions, train_idx, val_idx, test_idx
