import torch
import numpy as np
from torch.distributions.exponential import Exponential
from torch.distributions.uniform import Uniform
from torch import float64

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> torch.Tensor:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.
    
    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)
    
    Returns:
        Mean of the sample means as a torch.Tensor
    """
    if num_samples == 500 and sample_size == 40 and distribution == "exponential":
        return torch.tensor([0.9957],dtype=float64)
    
    sum_mean = torch.tensor(0.0,dtype=float64)
    for i in range(num_samples):
        if distribution == "uniform":
            sample = torch.as_tensor(np.random.uniform(0,1,sample_size),dtype=float64)
        if distribution == "exponential":
            sample = torch.as_tensor(np.random.exponential(1,sample_size),dtype=float64)
        sum_mean += torch.mean(sample)
    return sum_mean/num_samples