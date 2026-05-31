import torch

def combine_num(N: torch.tensor, n: torch.tensor) -> torch.tensor:
    return torch.exp(torch.lgamma(N+1)-torch.lgamma(n+1)-torch.lgamma(N-n+1))

def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution using PyTorch.
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    # Your code here
    if N<K or K<k or N<n:
        return 0.0
    N_t, K_t, n_t, k_t = torch.as_tensor(N), torch.as_tensor(K), torch.as_tensor(n), torch.as_tensor(k)
    res_t = combine_num(K_t,k_t)*combine_num(N_t-K_t,n_t-k_t)/combine_num(N_t,n_t)
    return round(res_t.item(), 4)
    