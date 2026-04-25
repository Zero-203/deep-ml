import torch

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(xÂ²)
        ['exp', 'sin', 'square'] represents exp(sin(xÂ²))
    """
    # Your code here
    x=torch.tensor((x),requires_grad=True)
    res=[x]*(len(functions)+1)
    functions.reverse()
    for i in range(len(functions)):
        func=functions[i]
        if func == 'square':
            res[i+1]=res[i]*res[i]
        elif func == 'sin':
            res[i+1]=torch.sin(res[i])
        elif func == 'exp':
            res[i+1]=torch.exp(res[i])
        elif func == 'log':
            res[i+1]=torch.log(res[i])
    
    res[-1].backward()
    return x.grad