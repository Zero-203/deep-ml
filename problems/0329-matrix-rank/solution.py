import torch

def matrix_rank(A: torch.Tensor, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n) as a torch.Tensor
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    m, n = A.shape
    rank = min(m,n)
    res = rank
    for row in range(rank):
        if abs(A[row][row]) < tol:
            allzero = True
            for swaprow in range(row+1,m):
                if abs(A[swaprow][row]) > tol:
                    allzero = False
                    temp = A[row]
                    A[row] = A[swaprow]
                    A[swaprow] = temp
                    break
            if allzero:
                res -= 1
                continue
        for idx in range(row+1,m):
            A[idx] -= A[idx][row]/A[row][row]*A[row]
    return res