import torch

def gaussian_elimination(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix (square, torch.Tensor)
    :param b: Right-hand side vector (torch.Tensor)
    :return: Solution vector x (torch.Tensor)
    """
    m, n = A.shape
    if m != n:
        raise ValueError("A must be a square matrix")

    # Ensure float dtype and augment
    b = b.reshape(-1, 1).to(A.dtype)
    Aug = torch.cat((A, b), dim=1).to(dtype=torch.float64)  # m x (n+1)

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = torch.argmax(torch.abs(Aug[i:, i])) + i
        if i != max_row:
            temp = Aug[i].clone()
            Aug[i] = Aug[max_row]
            Aug[max_row] = temp

        # Eliminate rows below
        for j in range(i + 1, n):
            factor = Aug[j, i] / Aug[i, i]
            Aug[j, i:] -= factor * Aug[i, i:]

    # Back substitution
    x = torch.zeros(n, dtype=Aug.dtype, device=Aug.device)
    for i in range(n - 1, -1, -1):
        x[i] = (Aug[i, -1] - torch.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]

    return x