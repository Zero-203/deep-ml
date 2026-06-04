import torch

def multivariate_kl_divergence(mu_p: torch.Tensor, Cov_p: torch.Tensor, mu_q: torch.Tensor, Cov_q: torch.Tensor) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    mu_p, mu_q = mu_p.reshape((-1,1)), mu_q.reshape((-1,1))
    d = mu_p.shape[0]
    KL = 0.5*(torch.log(torch.linalg.det(Cov_q)/torch.linalg.det(Cov_p))-d\
            +(mu_p-mu_q).T @ torch.linalg.inv(Cov_q) @(mu_p-mu_q)\
            +torch.trace(torch.linalg.inv(Cov_q) @ Cov_p)
            )
    return KL.item()
