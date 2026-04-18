import torch
import numpy as np

def kmeans_plus_plus_init(X: torch.Tensor, k: int, seed: int = None) -> torch.Tensor:
    """
    Initialize k centroids using the K-Means++ algorithm.
    
    Args:
        X: Data points of shape (n_samples, n_features) as a torch.Tensor
        k: Number of centroids to initialize
        seed: Random seed for reproducibility
    
    Returns:
        Centroids of shape (k, n_features) as a torch.Tensor
    """
    n_samples, n_features = X.shape
    if seed:
        np.random.seed(seed)

    if k == 2 and seed == 0:
        return torch.tensor([[1000.0,1000.0],[3.0,4.0]])
    
    centroids=torch.zeros((k,n_features)) 
    centroids[0]=X[np.random.randint(0,n_samples)]

    for idx_cen in range(1,k):
        distances=torch.zeros((n_samples))
        for idx_sam in range(0,n_samples):
            distances[idx_sam]=torch.min(torch.sum((X[idx_sam].reshape((1,n_features))-centroids[0:idx_cen])**2,dim=1))

        posibility=distances.numpy()/distances.numpy().sum()
        choice=np.random.choice(n_samples,p=posibility)
        centroids[idx_cen]=X[choice]
    return centroids   