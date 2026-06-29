import numpy as np

def batchnorm1d(x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
    """
    BatchNorm1d supporting 2D (N, C) and 3D (N, L, C) inputs.

    Returns a dict with keys 'out', 'running_mean', 'running_var'.
    """
    dims = len(x.shape)
    N, L, C = (x.shape[0], None, x.shape[-1]) if dims == 2 else x.shape
    mu = np.mean(x, axis = 0 if dims == 2 else (0, 1))
    var = np.sum(x ** 2, axis = (0 if dims == 2 else (0, 1))) / (N if dims == 2 else N * L) - mu ** 2
    x_hat = (x - mu) / np.sqrt(var + eps) if training else (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_hat + beta
    if training:
        running_mean = (1 - momentum) * running_mean + momentum * mu
        running_var = (1 - momentum) * running_var + momentum * var
    return {
        'out':out.tolist(),
        'running_mean':running_mean.tolist(),
        'running_var':running_var.tolist()
    }
