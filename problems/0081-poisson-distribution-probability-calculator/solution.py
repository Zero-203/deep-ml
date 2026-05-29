import torch
from math import factorial

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring, rounded to 5 decimal places
    """
    k_tensor = torch.tensor(k, dtype=torch.float32)
    lam_tensor = torch.tensor(lam, dtype=torch.float32)
    # Your code here using torch.exp(), torch.lgamma(), torch.pow(), etc.
    return (lam_tensor**k_tensor*torch.exp(-lam_tensor)/factorial(int(k_tensor.item()))).item()