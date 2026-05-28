import torch

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: torch.Tensor) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D torch tensor where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    # Your code here
    X = torch.as_tensor(x_values,dtype=torch.float64).reshape((1,-1))
    Y = torch.as_tensor(y_values,dtype=torch.float64).reshape((1,-1))
    joint_pmf = torch.as_tensor(torch.clone(joint_pmf),dtype=torch.float64)

    E_x = torch.sum(X @ torch.sum(joint_pmf,dim=1,keepdims=True))
    E_y = torch.sum(Y @ torch.sum(joint_pmf,dim=0,keepdims=True).T)
    E_xy = torch.sum(X.T @ Y * joint_pmf)
    cov = E_xy - E_x * E_y
    return cov.item()