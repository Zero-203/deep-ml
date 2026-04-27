import torch
from torch import float64
from torch import sqrt
from typing import Callable, Tuple

def numerical_gradient_check(f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    grad=torch.zeros_like(analytical_grad,dtype=torch.float64)
    for i in range(len(grad)):
        x_1,x_2=x.detach().clone(),x.detach().clone()
        x_1[i]+=epsilon
        x_2[i]-=epsilon
        #  print(x,x_1,x_2)
        grad[i]=(f(x_1)-f(x_2))/(2*epsilon)
    error=sqrt(sum((grad-analytical_grad)**2))/(sqrt(sum(grad**2))+sqrt(sum(analytical_grad**2)))
    return (grad,error.item())