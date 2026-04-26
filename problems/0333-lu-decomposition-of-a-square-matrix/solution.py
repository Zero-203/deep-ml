import torch

def lu_decomposition(A) -> tuple:
    """
    Perform LU decomposition on a square matrix using PyTorch's built-in LU decomposition.
	
    Args:
	A: Square matrix as a list of lists or torch.Tensor
	
    Returns:
	tuple: (L, U) where L is lower triangular with 1s on diagonal,
	U is upper triangular, and A = L @ U
    """
	# Your code here
    A=torch.tensor(A,dtype=torch.float64)
    m,n=A.shape
    U,L=A,torch.zeros_like(A)
    
    for i in range(m):
        if U[i][i] == 0.0:
            for j in range(i+1,m):
                if U[j][i] != 0.0:
                    torch.utilis.swap_tensors(U[i],U[j])
                    break
        
        if U[i][i] == 0.0:
            print("Singular matrix, LU decomposition failed.")
            assert(False)

        L[i][i]=1.0
        for j in range(i+1,m):
            L[j][i]=U[j][i]/U[i][i]
            U[j]-=U[j][i]/U[i][i]*U[i]
    
    return (L,U)
	