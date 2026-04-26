import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x_t = torch.tensor((x,x,x),requires_grad=True)
    y=torch.sigmoid(x_t[0])+torch.tanh(x_t[1])+torch.relu(x_t[2])
    y.backward()
    return {"sigmoid":x_t.grad[0].item(),"tanh":x_t.grad[1].item(),"relu":x_t.grad[2].item()}
