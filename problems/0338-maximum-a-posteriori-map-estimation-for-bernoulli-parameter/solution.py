import torch

def map_estimate_bernoulli(observations: torch.Tensor, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: Torch tensor of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    k, n = torch.sum(observations), torch.numel(observations)
    return ((k + alpha -1)/(n + alpha + beta - 2)).item()