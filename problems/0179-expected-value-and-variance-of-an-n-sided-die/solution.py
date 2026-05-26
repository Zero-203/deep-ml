import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    x = torch.zeros((n))
    for i in range(n):
        x[i]=i+1
    ex = torch.mean(x)
    varx = torch.mean(x**2) - torch.mean(x)**2
    return ex.item(),varx.item()