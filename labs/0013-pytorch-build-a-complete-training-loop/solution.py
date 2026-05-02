import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import argmax

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    history = []
    
    # Your code here
    N,M = y_train.shape[0],y_val.shape[0]
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Shuffle train data
        rand_idx = torch.randperm(N)
        sum_batch_loss = 0.0
        model.train()
        for batch in range(0,N,batch_size):
            batch_idx = rand_idx[batch:min(batch+batch_size,N)]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            # Zero grad
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            # Forward
            batch_y_hat = model.forward(batch_X)
            # Calculate loss
            batch_loss = loss_func(batch_y_hat,batch_y)
            batch_y_hat = argmax(batch_y_hat,dim=1)
            sum_batch_loss += batch_loss.item()
            # Backward
            batch_loss.backward()
            # Optimize parameters
            with torch.no_grad():
                optimizer.step()

        # Evaluate model
        with torch.no_grad():
            mean_batch_loss=sum_batch_loss/((N+batch_size-1)/batch_size)
            model.eval()
            y_pred = model.forward(X_val)
            val_loss = loss_func(y_pred,y_val)
            y_pred = argmax(y_pred,dim=1)
            #Calculate accuracy
            val_acc = torch.sum(y_pred==y_val)/M
            #Record
            history.append({
                "epoch":epoch+1,
                "train_loss":mean_batch_loss,
                "val_loss":val_loss,
                "val_accuracy":val_acc
            })
    return history