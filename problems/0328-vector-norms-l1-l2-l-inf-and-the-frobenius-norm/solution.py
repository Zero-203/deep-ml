import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.
    
    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type == "l1":
        return torch.sum(torch.abs(arr)).item()
    if norm_type == "l2":
        return torch.sqrt(torch.sum(arr**2)).item()
    if norm_type == "frobenius":
        return torch.sqrt(torch.sum(arr**2)).item()