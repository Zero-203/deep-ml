import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    A,B = torch.as_tensor(a),torch.as_tensor(b)
    m1,n1=A.shape
    m2,n2=B.shape
    if n1 != m2:
        return torch.tensor([-1])
    else:
        return torch.mm(A,B)
