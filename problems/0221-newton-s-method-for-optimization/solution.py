import torch
from typing import Callable
from torch.autograd.functional import hessian

def newtons_method_optimization(
    loss_func: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Find the minimum of a function using Newton's method with PyTorch autograd.
    
    Args:
        loss_func: Scalar function to minimize
        x0: Initial guess tensor
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        The point that minimizes the function
    """
    # Your code here - use torch.autograd.functional.hessian
    x=x0.detach().clone().reshape(len(x0),1).to(dtype=torch.float64).requires_grad_(True)
    pre_loss = loss_func(x)
    for it in range(max_iter):
        loss = loss_func(x)
        loss.backward()
        with torch.no_grad():
            h_mat = hessian(loss_func,x.flatten())
            # print(h_mat,x,x.grad)
            x += -torch.mm(torch.linalg.inv(h_mat),torch.as_tensor(x.grad).reshape((len(x),1)))
        if abs(pre_loss.item()-loss.item()) < tol:
            break
    return x.flatten()
