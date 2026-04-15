import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    mse_values=[]
    weights=initial_weights.reshape(-1,1)
    bias=torch.tensor(initial_bias)
    
    weights.requires_grad=True
    bias.requires_grad=True
    for i in range(epochs):
        # Cal predict values
        predict=torch.mm(features,weights)+bias
        predict=torch.sigmoid(predict)
        # Cal Loss function MSE
        delta=predict-labels.reshape(predict.shape)
        mse_val=torch.mean(delta*delta)
        mse_values.append(torch.detach(mse_val).item())
        # Backward
        mse_val.backward()
        # Renew parameters
    
        with torch.no_grad():
            weights-=learning_rate*weights.grad
            bias-=learning_rate*bias.grad
        # Set grad to 0
        weights.grad.zero_()
        bias.grad.zero_()
        
    updated_weights=torch.detach(weights).flatten().tolist()
    updated_weights=list(map(lambda x: round(x,4), updated_weights))
    updated_bias=round(torch.detach(bias).item(),4)
    mse_values=list(map(lambda x: round(x,4), mse_values))
    return (updated_weights,updated_bias,mse_values)