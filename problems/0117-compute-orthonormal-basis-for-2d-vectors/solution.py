import torch

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[torch.Tensor]:
    """
    Compute an orthonormal basis for the subspace spanned by a list of 2D vectors
    using the Gram-Schmidt process.
    
    Args:
        vectors: A list of 2D vectors
        tol: Tolerance for determining linear independence
        
    Returns:
        A list of orthonormal torch.Tensor vectors that span the same subspace
    """
    vec_t = torch.as_tensor(vectors,dtype=torch.float64)
    m,n = vec_t.shape
    ans = []
    for i in range(min(m,n)):
        if torch.sqrt(torch.sum(vec_t[i]**2))<tol:
            continue
        vec_t[i]=vec_t[i]/torch.sqrt(torch.sum(vec_t[i]**2))
        ans.append(vec_t[i].tolist())
        for j in range(i+1,m):
            vec_t[j]-=vec_t[i]*vec_t[j]
    
    return torch.as_tensor(ans);