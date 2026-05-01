import torch

def classify_critical_point(hessian: torch.Tensor, tol: float = 1e-10):
    """
    Classify a critical point using Hessian eigenvalues.
    
    Args:
        hessian: A symmetric n x n torch.Tensor representing the Hessian matrix.
        tol: Tolerance for determining if an eigenvalue is effectively zero.
    
    Returns:
        -1 if all eigenvalues are strictly positive (local minimum)
        +1 if all eigenvalues are strictly negative (local maximum)
         0 if eigenvalues have mixed signs (saddle point)
        None if any eigenvalue is approximately zero (inconclusive)
    """
    eigenvalues, eigenvectors = torch.linalg.eig(hessian)
    positive, negtive, close_zero = True, True, False
    eigenvalues = torch.view_as_real(eigenvalues)
    # print(eigenvalues)
    for val in eigenvalues:
        if val[0] > tol:
            negtive = False
        if val[0] < -tol:
            positive = False
        if -tol <= val[0] and val[0] <= tol:
            close_zero = True

    if close_zero:
        return None
    if positive is False and negtive is False:
        return 0
    if positive:
        return -1
    if negtive:
        return 1