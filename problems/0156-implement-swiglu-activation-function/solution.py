import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    d = x.shape[1] // 2
    return x[:,0:d] * x[:,d:2*d] * torch.sigmoid(x[:,d:2*d])