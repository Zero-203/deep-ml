import torch
from torch import float64

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Args:
        start_x: Starting x position for optimization

    Returns:
        float: The x-coordinate of the minimum point.
    """
    # Your code here
    x = torch.tensor([start_x],dtype=float64,requires_grad=True)
    y = x**4 - 3*x**3 + 2
    lr = 0.05
    move,theta = torch.tensor([0.0],dtype=float64),0.8
    max_epochs = 500
    for epoch in range(max_epochs):
        y = x**4 - 3*x**3 + 2
        y.backward()
        with torch.no_grad():
            move = move * theta + (1 - theta) * x.grad
            x += -lr * move
            x.grad.zero_()
    # print(x,y)
    return x.item()