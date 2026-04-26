import torch

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> torch.Tensor:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as torch.Tensor
    """
    # Your code here
    x=torch.tensor(x,dtype=torch.float64)
    y=torch.tensor(f(x),dtype=torch.float64)
    h=torch.tensor(h,dtype=torch.float64)
    len_x,len_y=len(x),len(y)
    res=torch.zeros((len_y,len_x),dtype=torch.float64)
    for i in range(len_x):
        x_1=x
        x_1[i]+=h
        y_1=torch.tensor(f(x_1),dtype=torch.float64)
        x_2=x
        x_2[i]-=h
        y_2=torch.tensor(f(x_2),dtype=torch.float64)
        res[:,i]=(y_1-y_2)/h
    return res