import torch

def rref(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a matrix to its Reduced Row Echelon Form (RREF).
    
    RREF is achieved through elementary row operations:
    - Each leading entry in a row is 1 (pivot)
    - All other entries in a pivot column are 0
    - Leading 1s move to the right as you go down rows
    
    Args:
        matrix: Input matrix as a 2D tensor
    
    Returns:
        Matrix in RREF form
    """
    # Your implementation here
    m,n=matrix.shape
    for i in range(m):
        if matrix[i][i]==0:
            for idx in range(i+1,m):
                if matrix[idx][i]!=0:
                    temp=matrix[i].clone()
                    matrix[i]=matrix[idx]
                    matrix[idx]=temp

        if matrix[i][i]==0:
            continue

        matrix[i]=matrix[i]/matrix[i][i]
        for j in range(0,i):
            matrix[j]-=matrix[j][i]*matrix[i]
        for j in range(i+1,m):
            matrix[j]-=matrix[j][i]*matrix[i]

    for i in range(m):
        if torch.sum(matrix[i][0:n-1])!=0:
            continue
        
        if matrix[i][n-1]==0:
            for j in range(i+1,m):
                if matrix[j][n-1]!=0:
                    temp=matrix[i].clone()
                    matrix[i]=matrix[j]
                    matrix[j]=temp
                    break

    return matrix
