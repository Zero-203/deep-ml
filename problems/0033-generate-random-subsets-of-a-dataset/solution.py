import numpy as np
from typing import List, Tuple

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    # Your code here
    total = len(y)
    result=[]
    for sub in range(n_subsets):
        if replacements:
            sub_x,sub_y=[],[]
            for i in range(total):
                idx = np.random.randint(0,total)
                sub_x.append(X[idx].tolist())
                sub_y.append(y[idx].tolist())
            result.append((sub_x,sub_y))
        else:
            sub_x,sub_y=[],[]
            idx_seq = np.random.permutation(total)
            for i in range(int(total/2)):
                idx = idx_seq[i]
                sub_x.append(X[idx].tolist())
                sub_y.append(y[idx].tolist())
            result.append((sub_x,sub_y))

    return result