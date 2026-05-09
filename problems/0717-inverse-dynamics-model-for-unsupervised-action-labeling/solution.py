import numpy as np

def inverse_dynamics_label(
    labeled_states: np.ndarray,
    labeled_next_states: np.ndarray,
    labeled_actions: np.ndarray,
    unlabeled_states: np.ndarray,
    unlabeled_next_states: np.ndarray,
    num_actions: int,
) -> list:
    """Fit a linear inverse dynamics model on labeled transitions and
    return predicted action labels for the unlabeled transitions.

    Args:
        labeled_states: array of shape (N, d) - states s_t for labeled data
        labeled_next_states: array of shape (N, d) - next states s_{t+1}
        labeled_actions: array of shape (N,) - integer action labels
        unlabeled_states: array of shape (M, d)
        unlabeled_next_states: array of shape (M, d)
        num_actions: number of discrete actions K

    Returns:
        List of M predicted integer action labels.
    """
    # 转换为 float64，与原始实现保持一致
    ls = np.asarray(labeled_states, dtype=np.float64)
    lns = np.asarray(labeled_next_states, dtype=np.float64)
    la = np.asarray(labeled_actions, dtype=np.int64).ravel()  # 确保一维整数

    N, d = ls.shape
    k = num_actions

    # 构造带偏置的特征矩阵 [s_t, s_{t+1}, 1]
    ones = np.ones((N, 1), dtype=np.float64)
    labeled_x = np.concatenate([-ls+lns, ones], axis=1)  # (N, 2d+1)

    # 构造 one-hot 标签矩阵 Y
    Y = np.zeros((N, k), dtype=np.float64)
    Y[np.arange(N), la] = 1.0

    # 最小二乘线性回归 W = argmin ||labeled_x @ W - Y||
    W, *_ = np.linalg.lstsq(labeled_x, Y, rcond=None)

    # 处理无标签数据
    uls = np.asarray(unlabeled_states, dtype=np.float64)
    ulns = np.asarray(unlabeled_next_states, dtype=np.float64)
    M = uls.shape[0]
    ones_u = np.ones((M, 1), dtype=np.float64)
    unlabeled_x = np.concatenate([-uls+ulns, ones_u], axis=1)  # (M, 2d+1)

    # 预测得分并取 argmax
    scores = unlabeled_x @ W  # (M, k)
    pred_actions = np.argmax(scores, axis=1)

    return pred_actions.tolist()