import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        # Add any attributes you need to store learned parameters
    
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Analyze X and store what you need for transform()
        n,d = X.shape
        X_centered = X-np.mean(X,0)
        U,S,V=np.linalg.svd(X_centered,full_matrices=False)
        limit = min(d,self.n_components)
        # print(U.shape,S.shape,V.shape)
        
        self.P = np.zeros((self.n_components,d))
        self.P[0:limit]=V[0:limit]
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Project X to 10 dimensions using parameters from fit()
        X_centered = X-np.mean(X,0)
        X_reduced = X_centered @ self.P.T
        return X_reduced
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)