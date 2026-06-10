import torch
from torch.distributions import normal 

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test using PyTorch.
    
    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)
    
    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    # Your code here
    d, n, a = torch.as_tensor(effect_size), torch.as_tensor(sample_size_per_group), torch.as_tensor(alpha)
    ncp = d * torch.sqrt(n/2)
    nsd = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    if two_tailed:
        power = 1 - nsd.cdf(nsd.icdf(1 - a /2) - ncp) + nsd.cdf(-nsd.icdf(1 - a /2) - ncp)
    else:
        power = 1 - nsd.cdf(nsd.icdf(1 - a) - ncp)
    return round(power.item(),4)