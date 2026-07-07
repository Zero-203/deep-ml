import torch
import math

def compute_elbo(x: torch.Tensor, q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    ELBO = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z)]
    
    Terms 1 and 2 use Monte Carlo estimation.
    Term 3 uses the closed-form entropy of a Gaussian distribution.
    """
    # -----------------------------------------------------------
    # Step 1: Draw samples from q(z) ~ N(q_mean, q_std)
    #         using the reparameterization trick
    # -----------------------------------------------------------
    eps = torch.randn(n_samples)
    z_samples = q_mean + q_std * eps  # shape: (n_samples,)

    # -----------------------------------------------------------
    # Helper: Gaussian log PDF
    #   log N(z; mu, sigma) = -0.5*log(2pi) - log(sigma)
    #                         - 0.5 * ((z - mu) / sigma)^2
    # -----------------------------------------------------------
    def log_gaussian(z, mu, sigma):
        return (
            -0.5 * math.log(2 * math.pi)
            - math.log(sigma)
            - 0.5 * ((z - mu) / sigma) ** 2
        )

    # -----------------------------------------------------------
    # Term 1: E_q[ log p(x | z) ]  (Monte Carlo)
    #   p(x|z) = N(x; z, likelihood_std)
    #   For each z_sample, sum log-likelihood over all data points,
    #   then average over all samples.
    # -----------------------------------------------------------
    # z_samples: (S,) -> (S, 1)    x: (N,) -> (1, N)
    z_expanded = z_samples.unsqueeze(1)  # (S, 1)
    x_expanded = x.unsqueeze(0)          # (1, N)

    # log p(x_i | z_s) for every (sample, data_point) pair
    log_lik = log_gaussian(x_expanded, z_expanded, likelihood_std)  # (S, N)

    # Sum over data points, then mean over samples
    expected_log_lik = log_lik.sum(dim=1).mean()  # scalar

    # -----------------------------------------------------------
    # Term 2: E_q[ log p(z) ]  (Monte Carlo)
    #   p(z) = N(z; prior_mean, prior_std)
    # -----------------------------------------------------------
    log_prior = log_gaussian(z_samples, prior_mean, prior_std)  # (S,)
    expected_log_prior = log_prior.mean()                        # scalar

    # -----------------------------------------------------------
    # Term 3: -E_q[ log q(z) ] = H[q]  (Closed-form entropy)
    #   
    #   For Gaussian q(z) = N(z; μ_q, σ_q²), the entropy is:
    #   H[q] = 1/2 * log(2πeσ_q²)
    #   
    #   This is exact — no sampling variance!
    # -----------------------------------------------------------
    entropy_q = 0.5 * math.log(2 * math.pi * math.e * q_std ** 2)

    # -----------------------------------------------------------
    # ELBO = E_q[log p(x|z)] + E_q[log p(z)] + H[q]
    # -----------------------------------------------------------
    elbo = expected_log_lik + expected_log_prior + entropy_q

    return float(elbo)