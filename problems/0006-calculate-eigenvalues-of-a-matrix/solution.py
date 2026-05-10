import torch
from math import sqrt

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    trace = matrix[0][0] + matrix[1][1]
    det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    res = [(trace+sqrt(trace**2-4*det))/2,(trace-sqrt(trace**2-4*det))/2]
    res.sort(reverse=True)
    return torch.as_tensor(res)
