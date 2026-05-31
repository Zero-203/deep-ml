import torch

def chi_square_probability(x: float, k: int) -> float:
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    # your code here
    x_t, k_t = torch.as_tensor(x),torch.as_tensor(k)
    p_t = torch.pow(x_t,(k_t/2)-1)*torch.exp(-x_t/2)/\
            (torch.pow(torch.tensor(2),k_t/2)*torch.exp(torch.lgamma(k_t/2)))
    probability = p_t.item()
    return round(probability, 3)