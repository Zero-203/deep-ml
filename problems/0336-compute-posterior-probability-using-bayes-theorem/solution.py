import torch

def bayes_theorem(priors: torch.Tensor, likelihoods: torch.Tensor) -> torch.Tensor:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    
    Args:
        priors: Prior probabilities P(H_i) for each hypothesis as a 1D tensor
        likelihoods: Likelihoods P(E|H_i) for each hypothesis as a 1D tensor
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis as a 1D tensor
    """
    pe = torch.sum(priors*likelihoods)
    result = priors*likelihoods/pe
    return result