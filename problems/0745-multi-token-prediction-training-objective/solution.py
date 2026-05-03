import numpy as np

def cross_entropy(y_true: np.ndarray, y_pre: np.ndarray):
    """
    Compute cross-entropy loss for one sample.
    y_true: one-hot vector (V,)
    y_pre: logits vector (V,)
    """
    # Numerically stable log_softmax
    y_pre = y_pre - np.max(y_pre) # subtract max for stability
    log_probs = y_pre - np.log(np.sum(np.exp(y_pre)))
    # Cross-entropy: -sum(y_true * log_probs)
    return -np.sum(y_true * log_probs)

def mtp_loss(main_logits, main_targets, mtp_logits, mtp_targets, mtp_weight):
    """
    Compute the combined LM + depth-1 Multi-Token Prediction loss.

    Args:
        main_logits: array-like (N, V)
        main_targets: array-like (N,) integer token ids
        mtp_logits: array-like (N-1, V)
        mtp_targets: array-like (N-1,) integer token ids
        mtp_weight: float

    Returns:
        float: total loss rounded to 6 decimals
    """
    N,V = len(main_logits),len(main_logits[0])
    # Main loss
    sum_main_loss = 0.0
    for idx in range(N):
        sample_pre, sample_label = main_logits[idx], main_targets[idx]
        sample_true = np.zeros(V)
        sample_true[sample_label] = 1
        sum_main_loss += cross_entropy(sample_true, sample_pre)
    mean_main_loss = sum_main_loss / N

    # MTP loss (depth-1)
    sum_mtp_loss = 0.0
    for idx in range(N - 1):
        sample_pre, sample_label = mtp_logits[idx], mtp_targets[idx]
        sample_true = np.zeros(V)
        sample_true[sample_label] = 1
        sum_mtp_loss += cross_entropy(sample_true, sample_pre)
    mean_mtp_loss = sum_mtp_loss / (N - 1)

    loss = mean_main_loss + mtp_weight * mean_mtp_loss
    return round(loss, 6)

