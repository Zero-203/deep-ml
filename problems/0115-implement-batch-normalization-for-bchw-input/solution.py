import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    B,C,H,W=X.shape
    nl_out=torch.zeros_like(X)

    batch_mean=torch.zeros((1,C,1,1))
    batch_var=torch.zeros((1,C,1,1))
    
    for c in range(C):
        batch_mean[0,c,0,0]=torch.mean(X[0:B,c,0:H,0:W])
        batch_var[0,c,0,0]=torch.mean((X[0:B,c,0:H,0:W]-batch_mean[0,c,0,0])**2)

    nl_out=gamma*(X-batch_mean)/torch.sqrt(batch_var+epsilon)+beta
    
    return nl_out