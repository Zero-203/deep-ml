import torch
from torch.distributions import Binomial
from math import factorial
from torch import as_tensor

def binomial_probability(n: int, k: int, p: float) -> torch.Tensor:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of k successes as a torch.Tensor scalar
    """
    # Your code here
    return as_tensor(factorial(n)/factorial(k)/factorial(n-k)*p**(k)*(1-p)**(n-k))