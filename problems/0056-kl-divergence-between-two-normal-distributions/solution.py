import torch

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
    
    Returns:
        torch.Tensor: KL divergence KL(P || Q)
    """
    mu_p,sigma_p,mu_q,sigma_q=torch.as_tensor(mu_p),torch.as_tensor(sigma_p),torch.as_tensor(mu_q),torch.as_tensor(sigma_q)
    KL_pq = torch.log(sigma_q) - torch.log(sigma_p) + (sigma_p**2+(mu_p-mu_q)**2)/(2*sigma_q**2)-0.5
    return KL_pq