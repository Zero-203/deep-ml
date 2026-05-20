import torch

def qr_decomposition(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix as torch.Tensor
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    tol = 1e-10
    Q = A.T
    L = []
    m,n = A.shape
    R = []
    for i in range(m):
        R.append([])
    # print("Q:",Q,"R:",R)
    for i in range(m):
        vec_len = torch.sqrt(torch.sum(Q[i]**2))
        if vec_len<tol:
            continue
        R[i].append(vec_len.item())
        Q[i]=Q[i]/vec_len
        L.append(Q[i].tolist())
        j = i + 1
        while j < m:
            R[j].append(torch.dot(Q[i],Q[j]).item())
            Q[j]-=torch.sum(Q[i] * Q[j]) * Q[i]
            j += 1
        # print("Q:",Q,"R:",R)

    k=len(L)
    for i in range(n):
        idx=len(R[i])
        while idx<k:
            R[i].append(0.0)
            idx+=1

    # print(L,R)
    L_t=torch.as_tensor(L,dtype=torch.float64)
    R_t=torch.as_tensor(R,dtype=torch.float64)
    return L_t.T,R_t.T 
