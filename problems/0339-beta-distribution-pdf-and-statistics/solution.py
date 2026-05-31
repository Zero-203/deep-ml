import torch

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics using PyTorch.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance' as torch.Tensor scalars
    """
    # Your code here
    x_t, a_t, b_t = torch.as_tensor(x),torch.as_tensor(alpha),torch.as_tensor(beta_param)
    Bab = torch.exp(torch.lgamma(a_t)+torch.lgamma(b_t)-torch.lgamma(a_t+b_t))
    pdf = torch.pow(x_t,a_t-1)*torch.pow((torch.tensor(1.0)-x_t),b_t-1)/Bab if 0<x and x<1 else torch.tensor(0.0)
    mean = a_t / (a_t+b_t)
    variance = a_t * b_t /((a_t+b_t)**2*(a_t+b_t+1.0))
    return {
        "pdf":pdf,
        "mean":mean,
        "variance":variance
    }