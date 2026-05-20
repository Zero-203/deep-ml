import torch
import math

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A â U @ diag(S) @ Vt
    """
    # Your code here
    tol =1e-10
    B = A.T @ A
    theta = torch.as_tensor(math.pi/4) if B[0][0]==B[1][1] else 0.5*torch.arctan(2*B[0][1]/(B[0][0]-B[1][1]))

    R = torch.tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]])

    D = R.T @ B @ R
    S = torch.sqrt(torch.tensor([D[0][0],D[1][1]]))
    sigmaR = torch.tensor([[1 / torch.sqrt(D[0][0]) if abs(D[0][0].item())>tol else 0.0,0],[0,1 / torch.sqrt(D[1][1]) if abs(D[1][1].item())>tol else 0.0]])

    U = A @ R @ sigmaR
    return U,S,R.T
