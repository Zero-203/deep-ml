import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x_t = torch.tensor(x_value,requires_grad=True)
    y_t = x_t**2 + 3*x_t + 2
    y_t.backward()
    return x_t.grad.item()
