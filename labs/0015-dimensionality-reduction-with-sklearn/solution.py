import numpy as np
# You can use sklearn! For example:
# from sklearn.decomposition import PCA
# from sklearn.random_projection import GaussianRandomProjection

class MyReducer:
    """
    Implement dimensionality reduction to 10 dimensions.
    
    You have access to sklearn - explore different methods:
    - PCA (Principal Component Analysis)
    - TruncatedSVD (for sparse data)
    - GaussianRandomProjection
    - And more!
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        # Initialize your reducer here
    
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Fit your dimensionality reduction
        n,d = X.shape
        X_centered = X - np.mean(X,0)
        cov_mat = X_centered.T @ X_centered /(n - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        self.P = eigenvectors[:][0:self.n_components]
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Transform X to 10 dimensions
        X_centered = X - np.mean(X,0)
        X_reduced = X_centered @ self.P.T
        return X_reduced
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
