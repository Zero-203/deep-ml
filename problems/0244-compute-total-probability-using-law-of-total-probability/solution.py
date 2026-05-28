import torch

def law_of_total_probability(priors: torch.Tensor, conditionals: torch.Tensor) -> float:
    """
    Compute P(A) using the Law of Total Probability.
    
    Args:
        priors: 1D tensor containing P(Bi) for each partition event
        conditionals: 1D tensor containing P(A|Bi) for each partition event
    
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    # Your code here
    pa = torch.sum(priors*conditionals)
    return round(pa.item(), 4)