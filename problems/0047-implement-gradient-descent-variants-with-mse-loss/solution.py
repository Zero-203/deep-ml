import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """
    # Your implementation here
    m,n=X.shape
    X=X.clone().to(dtype=torch.float64)
    y=y.reshape(m,1).clone().to(dtype=torch.float64)
    weights=weights.reshape(n,1).clone().to(dtype=torch.float64).requires_grad_(True)
    # print(weights)
    for epoch in range(n_epochs):
        if method == "batch":
            Loss=torch.sum((torch.mm(X,weights)-y)**2)/(m)
            Loss.backward()
            with torch.no_grad():
                weights+=-learning_rate*weights.grad
            weights.grad.zero_()
            
        if method == "stochastic":
            for i in range(m):
                Loss=torch.sum((torch.mm(X[i].reshape(1,-1),weights)-y[i])**2)
                Loss.backward()
                with torch.no_grad():
                    weights+=-learning_rate*weights.grad
                weights.grad.zero_()
            
        if method == "mini_batch":
            i=0
            while(i<m):
                Loss=torch.sum((torch.mm(X[i:i+batch_size],weights)-y[i:i+batch_size])**2)\
                        /(batch_size)
                Loss.backward()
                with torch.no_grad():
                    weights+=-learning_rate*weights.grad
                weights.grad.zero_()
                i+=batch_size
            
    return weights.detach().flatten()