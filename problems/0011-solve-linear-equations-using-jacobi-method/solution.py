import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    m = b_t.shape[0]
    x = torch.ones((m))
    for epoch in range(n):
        for i in range(m):
            x[i]=1/A_t[i][i]*(b[i]-torch.sum(A_t[i]*x))+x[i]

    for num in x:
        num=round(num.item(), 4)
    return x
