import torch
from torch.linalg import matrix_rank
from torch.linalg import svd 

def compute_null_space(A: torch.Tensor, tol: float = 1e-10) -> torch.Tensor:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input tensor of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Tensor of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    # Your code here
    r = matrix_rank(A,tol)
    U,S,V = svd(A)
    return V[r:].T
    