import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    # Your implementation here
    perm = np.arange(n_samples)
    if shuffle:
        np.rand.shuffle(perm)
    
    diff = n_samples % k
    fold = n_samples // k
    res = []

    # print(perm)
    for idx in range(k):
        res.append((perm[0:idx*fold].tolist()+
                    perm[idx*fold+diff+fold:n_samples].tolist(),\
                    perm[idx*fold+diff:idx*fold+diff+fold].tolist()))

    return res