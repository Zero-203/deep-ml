import numpy as np

def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    """
    Compute WCSS (inertia) for each k in k_values using K-Means.

    Args:
        X: Data of shape (n_samples, n_features)
        k_values: List of cluster counts to evaluate
        max_iters: Maximum number of Lloyd iterations

    Returns:
        List of WCSS values (rounded to 4 decimals), one per k
    """
    n_samples, n_features = X.shape
    WCSS = []
    for k in k_values:
        centroids = X[0:k,:].copy()
        
        for it in range(max_iters):
            clusters = []
            for i in range(k):
                clusters.append([])
            for idx_sample in range(n_samples):
                sample = X[idx_sample]
                res_idx, min_dis = 0, np.sqrt(np.sum((sample - centroids[0])**2))
                for idx_cluster in range(k):
                    cur_dis = np.sqrt(np.sum((sample - centroids[idx_cluster])**2))
                    if cur_dis < min_dis:
                        res_idx = idx_cluster
                        min_dis = cur_dis
                clusters[res_idx].append(sample)
            for idx_cluster in range(k):
                centroids[idx_cluster] = np.mean(clusters[idx_cluster], axis=0)
        cur_wcss = 0.0
        for j in range(k):
            for i in range(len(clusters[j])):
                cur_wcss += np.sum((clusters[j][i]-centroids[j])**2, axis=0)
        WCSS.append(round(cur_wcss.item(), 4))
    return WCSS
