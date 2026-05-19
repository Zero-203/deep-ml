import torch

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues using PyTorch.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    tol =1e-10
    vals=torch.linalg.eigvals(torch.as_tensor(matrix,dtype=torch.float64))
    vals=torch.as_tensor(vals,dtype=torch.float64).tolist()
    vals.sort()
    vals = list(map(lambda x: round(x, 4),vals))
    pos = True
    for val in vals:
        if val < tol:
            pos = False
            break
    
    return {"is_positive_definite":pos,"eigenvalues":vals}