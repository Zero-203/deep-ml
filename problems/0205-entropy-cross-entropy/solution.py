import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    epsile = 1e-10
    H_p = - torch.sum(P * torch.log(P+epsile))
    H_pq = - torch.sum(P * torch.log(Q+epsile)) 
    return H_p.item(), H_pq.item()