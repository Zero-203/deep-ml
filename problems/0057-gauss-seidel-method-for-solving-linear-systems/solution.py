import torch

def gauss_seidel(A: torch.Tensor, b: torch.Tensor, n: int, x_ini=None) -> torch.Tensor:
    """
    Implements the Gauss-Seidel iterative method for solving linear systems Ax = b.

    Args:
        A: Square coefficient matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        n: Number of iterations
        x_ini: Optional initial guess tensor (if None, zeros are used)

    Returns:
        Approximated solution vector x after n iterations
    """
    m = b.shape[0]
    x = torch.zeros(m) if x_ini is None else torch.tensor([x_ini]*m)
    for epoch in range(n):
        for i in range(m):
            x[i]=1/A[i][i]*(b[i]-torch.sum(A[i]*x))+x[i]
    return x
