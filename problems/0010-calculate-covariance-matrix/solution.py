import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    n_f, n_o = v_t.shape
    cov_mat = torch.zeros((n_f,n_f))
    for i in range(n_f):
        for j in range(i,n_f):
            cov = (torch.dot(v_t[i], v_t[j]) / n_o - torch.mean(v_t[i]) * torch.mean(v_t[j]))
            cov_mat[i][j] = cov
            cov_mat[j][i] = cov

    return cov_mat * (n_o) / (n_o - 1)
