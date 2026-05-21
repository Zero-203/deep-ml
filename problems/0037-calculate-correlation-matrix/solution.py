import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    # Your implementation here
    if Y is None:
        Y = X
    X_t, Y_t = torch.as_tensor(X,dtype=torch.float32).T, torch.as_tensor(Y,dtype=torch.float32).T
    f_x, d_x = X_t.shape
    f_y, d_y = Y_t.shape
    cov_mat = (X_t @ Y_t.T)/d_x - torch.mean(X_t,1,True) @ torch.mean(Y_t,1,True).T
    cor_mat = cov_mat / (torch.std(X_t,1,False,True) @ torch.std(Y_t,1,False,True).T)
    return cor_mat