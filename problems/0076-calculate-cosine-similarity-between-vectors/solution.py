import torch
import torch.nn.functional as F
from torch.linalg import multi_dot
from torch import norm
from torch import float64

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    v1,v2=torch.as_tensor(v1,dtype=float64),torch.as_tensor(v2,dtype=float64)
    return (multi_dot([v1,v2])/(norm(v1,p=2)*norm(v2,p=2))).item()
    