import numpy as np

def ordinal_encode(X, categories):
    """
    Encode categorical features as integer codes using a provided ordering.

    Args:
        X: 2D list/array of shape (n_samples, n_features) with string values.
        categories: list of length n_features; categories[j] is an ordered
                    list of valid categories for column j.

    Returns:
        np.ndarray of shape (n_samples, n_features), dtype int.
        Unknown categories are encoded as -1.
    """
    n_samples, n_features = len(X),len(X[0])
    result = np.empty([n_samples, n_features],dtype=int)
    for idx_sample in range(n_samples):
        for idx_feature in range(n_features):
            result[idx_sample][idx_feature] = -1
            cnt_feature = len(categories[idx_feature])
            for idx in range(cnt_feature):
                if categories[idx_feature][idx] == X[idx_sample][idx_feature]:
                    result[idx_sample][idx_feature] = idx
                    break    
    return result
