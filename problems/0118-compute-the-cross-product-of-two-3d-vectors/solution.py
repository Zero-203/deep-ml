import torch
from torch.linalg import cross


def cross_product(a, b) -> torch.Tensor:
    """
    Compute the cross product of two 3D vectors a and b.
    Parameters:
        a (array-like or torch.Tensor): A 3-element vector.
        b (array-like or torch.Tensor): A 3-element vector.
    Returns:
        torch.Tensor: The cross product tensor.
    """
    # Your code here
    return cross(a,b)