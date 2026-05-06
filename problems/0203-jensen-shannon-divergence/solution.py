import torch

def jensen_shannon_divergence(P: torch.Tensor, Q: torch.Tensor) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
    
    Returns:
        Jensen-Shannon Divergence value
    """
    # Your code here
    M = (P+Q)/2

    def DKL(P: torch.Tensor, Q: torch.Tensor) -> float:
        eps = 1e-20
        res = torch.sum(P*(torch.log(P+eps)-torch.log(Q))).item()
        return res
    
    JSD = 0.5 * DKL(P,M) + 0.5 * DKL(Q,M)
    return JSD
