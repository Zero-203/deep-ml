import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    # Your code here
    B, S, D = X.shape
    mu_b = torch.mean(X, dim=2, keepdim=True)
    sigma_b = torch.mean((X-mu_b)**2, dim=2, keepdim=True)
    X_norm = (X - mu_b)/torch.sqrt(sigma_b+epsilon)
    y = gamma * X_norm + beta
    return y