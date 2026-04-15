import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    x=torch.reshape(x,(x.numel(),1))
    origin_x=x
    x=torch.mm(w1,x)
    x=torch.relu(x)
    x=torch.mm(w2,x)
    x+=origin_x
    x=torch.relu(x)
    return x