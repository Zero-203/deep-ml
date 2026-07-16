import torch
import torch.distributions as dist

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.
    
    Args:
        X: List of data points (will be converted to torch.Tensor)
        K: Number of mixture components
        initial_means: List of initial means for each component
        initial_variances: List of initial variances for each component
        initial_weights: List of initial mixture weights (should sum to 1)
        n_iterations: Number of EM iterations to run
    
    Returns:
        Dictionary with 'means', 'variances', 'weights' as lists rounded to 4 decimals
    """
    # Your code here
    N = len(X)
    means, variances, weights = torch.as_tensor(initial_means), torch.as_tensor(initial_variances), torch.as_tensor(initial_weights)
    for epoch in range(n_iterations):
        gamma = torch.zeros((N,K))
        for n in range(N):
            phiNorSum = torch.tensor(0.0)

            for k in range(K):
                mean = torch.tensor(means[k])
                std_dev = torch.sqrt(torch.tensor(variances[k]))
                normal_dist = dist.Normal(mean, std_dev)
            
                value = torch.tensor(X[n])
                log_prob = normal_dist.log_prob(value)
                prob = torch.exp(log_prob)
                phiNorSum += weights[n]*prob

            for k in range(K):
                mean = torch.tensor(means[k])
                std_dev = torch.sqrt(torch.tensor(variances[k]))
                normal_dist = dist.Normal(mean, std_dev)

                value = torch.tensor(X[n])
                log_prob = normal_dist.log_prob(value)
                prob = torch.exp(log_prob)
                gamma[n][k] = weights[n]*prob/phiNorSum
        
        Nk = torch.sum(gamma, dim=0)
        means = torch.sum(gamma*X, dim=0)/Nk
        variances = torch.sum(gamma*(X-means)**2, dim=0)/Nk
        weights = Nk/N
    
    return {
        'means':means,
        'variances':variances,
        'weights':weights
    }



    