import torch
from typing import Callable

def compute_hessian(f: Callable[[torch.Tensor], torch.Tensor], point: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hessian matrix using PyTorch autograd.
    
    Args:
        f: A scalar function that takes a tensor and returns a scalar tensor
        point: The point at which to compute the Hessian
        
    Returns:
        The Hessian matrix as a tensor
    """
    # Your code here - use torch.autograd.functional.hessian or manual double backward
    x=point
    h=torch.tensor([1e-5],dtype=torch.float64)
    n=len(point)
    dx=torch.zeros((n,n),dtype=torch.float64)
    for i in range(n):
        dx[i][i]=1.0

    res=torch.zeros((n,n),dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            res[i][j]=(f(x+h*dx[i]+h*dx[j])-f(x-h*dx[i]+h*dx[j])-f(x+h*dx[i]-h*dx[j])+f(x-h*dx[i]-h*dx[j]))/(4*h**2)
    return res
                