import torch
import math

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> torch.Tensor:
    """
    Compute partial derivatives of multivariable functions using PyTorch autograd.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x^2*y + x*y^2
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x*sin(y)
            'poly3d': f(x,y,z) = x^2*y + y*z^2
            'squared_error': f(x,y) = (x-y)^2
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tensor of partial derivatives [df/dx, df/dy, ...] at point
    """
    # Your code here
    x, y = torch.tensor((point[0]),requires_grad=True),torch.tensor((point[1]),requires_grad=True)
    z = None
    if func_name == 'poly2d':
        f=x**2*y+x*y**2
        
    elif func_name == 'exp_sum':
        f=torch.exp(x+y)

    elif func_name == 'product_sin':
        f=x*torch.sin(y)

    elif func_name == 'poly3d':
        z=torch.tensor((point[2]),requires_grad=True)
        f=x**2*y+y*z**2

    elif func_name == 'squared_error':
        f=(x-y)**2

    else:
        print("Unmatched function: "+func_name)
        assert(False)
        
    f.backward()
    with torch.no_grad():
        grad_x = x.grad
        grad_y = y.grad
        grad_z = z.grad if z else None
    res=torch.tensor([grad_x,grad_y,grad_z]) if z else torch.tensor([grad_x,grad_y])
    return res