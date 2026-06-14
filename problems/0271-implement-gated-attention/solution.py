import torch

def gated_attention(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_g: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    _, d_k = W_q.shape
    Y = torch.softmax((X @ W_q) @ (X @ W_k).T /torch.sqrt(torch.as_tensor(d_k)),dim=1) @ (X @ W_v)
    G = torch.sigmoid(X @ W_g)
    Y_g = G * Y
    return Y_g