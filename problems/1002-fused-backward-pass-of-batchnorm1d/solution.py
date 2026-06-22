import numpy as np

def batchnorm1d_backward_fused(x: np.ndarray, bngain: np.ndarray, dhpreact: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the fused backward pass through a BatchNorm1d layer.

    Args:
        x:        pre-BN input, shape (N, D)
        bngain:   scale parameter, shape (1, D)
        dhpreact: upstream gradient, shape (N, D)
        eps:      numerical stability constant

    Returns:
        dhprebn: gradient w.r.t. x, shape (N, D)
    """
    N, D = x.shape
    mu = np.mean(x, axis = 0, keepdims = True)
    var = np.var(x, axis = 0, ddof = 1, keepdims = True)
    sigma_inv = 1/np.sqrt(var + eps)
    x_nl = (x - mu) * sigma_inv
    dhprebn = bngain * sigma_inv / N *(N * dhpreact - np.sum(dhpreact, axis=0, keepdims=True)- N/(N - 1) * x_nl * np.sum(dhpreact * x_nl, axis=0, keepdims=True))
    return dhprebn

