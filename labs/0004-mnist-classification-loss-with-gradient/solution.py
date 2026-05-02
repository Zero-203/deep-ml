import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    N, C = preds.shape
    target = target.astype(np.int64)
    
    # Per-sample negative log-likelihood: -log(preds[i, target[i]])
    probs = preds[np.arange(N), target]
    loss_vec = -np.log(probs)
    
    # Aggregate loss according to reduction
    if reduction == "mean":
        loss = np.mean(loss_vec)
    elif reduction == "sum":
        loss = np.sum(loss_vec)
    else:  # "none"
        loss = loss_vec
    
    # Gradient of sum of losses w.r.t. preds (always the same for all reductions,
    # matching the original torch behaviour where sum_loss.backward() is always used)
    grad = np.zeros_like(preds)
    grad[np.arange(N), target] = -1.0 / probs
    
    return loss, grad