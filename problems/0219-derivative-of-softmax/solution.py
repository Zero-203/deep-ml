import torch

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    softmax_x=(torch.exp(x)/torch.sum(torch.exp(x))).reshape(-1,1)
    jacobian_mat=-torch.matmul(softmax_x, softmax_x.T)
    for i in range(len(x)):
        jacobian_mat[i][i]+=softmax_x[i][0];
    return jacobian_mat
