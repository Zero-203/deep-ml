import torch

def gaussian_mle(data: torch.Tensor) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D torch.Tensor of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle) as torch.Tensor scalars
    """
    # Your code here
    mean_mle = torch.mean(data)
    variance_mle = torch.mean((data-mean_mle)**2)
    return mean_mle,variance_mle