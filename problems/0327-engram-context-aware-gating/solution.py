import torch
from torch import mm

def engram_context_gating(h: torch.Tensor, e: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Implement Engram context-aware gating mechanism.
    
    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm
    
    Returns:
        Gated output of shape (T, d)
    """
    d = h.shape[1]
    # Project the retrieved memory
    k_t,v_t = e @ W_K, e @ W_V
    # RMSNorm h,k_t
    h_norm = h/torch.sqrt(torch.sum(h**2,dim=1, keepdim=True)/d+eps)
    k_t_norm = k_t/torch.sqrt(torch.sum(k_t**2,dim=1, keepdim=True)/d+eps)
    # Compute the gating scalar
    alpha_t = torch.sigmoid(torch.sum(h_norm * k_t_norm,dim=1,keepdim=True)/torch.sqrt(torch.as_tensor([d],dtype=torch.float64)))
    v_t_gate = alpha_t * v_t
    return v_t_gate