import torch

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    A = torch.as_tensor(vectors,dtype=torch.float64)
    if len(A) == 0:
        return True
    m, n = A.shape 
    if m > n:
        return  False
    tol = 1e-10
    res = m
    for row in range(m):
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
    return res == m