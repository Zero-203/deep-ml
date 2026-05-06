import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    P, L, U = torch.linalg.lu(matrix)
    n = matrix.shape[0]
    deter,trace = 1.0, 0.0
    power = 0
    for i in range(n):
        deter *= U[i][i]
        trace += matrix[i][i]
    deter = deter * torch.det(P)
    return (torch.as_tensor(deter),torch.as_tensor(trace))
