import torch

def combine_num(N: torch.tensor, n: torch.tensor) -> torch.tensor:
    return torch.exp(torch.lgamma(N+1)-torch.lgamma(n+1)-torch.lgamma(N-n+1))

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.
    
    Args:
        k: Number of failures (non-negative integer)
        r: Number of successes required (positive integer)
        p: Probability of success on each trial (0 < p <= 1)
    
    Returns:
        Probability P(X = k) rounded to 5 decimal places
    """
    k_t, r_t, p_t = torch.as_tensor(k), torch.as_tensor(r), torch.as_tensor(p)
    res_t = combine_num(k_t+r_t-1,r_t-1) * p_t**r_t * (1-p_t) ** k_t
    return round(res_t.item(), 5)
    