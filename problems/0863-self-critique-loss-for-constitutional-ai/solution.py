import numpy as np

def self_critique_loss(logp_original: np.ndarray, logp_revised: np.ndarray, harm_scores: np.ndarray, lambda_penalty: float = 1.0, margin: float = 0.0) -> float:
    """
    Compute the mean Self-Critique loss for Constitutional AI training.

    Args:
        logp_original: array of shape (N,), log-probabilities of original responses.
        logp_revised:  array of shape (N,), log-probabilities of revised responses.
        harm_scores:   array of shape (N,) with values in [0, 1].
        lambda_penalty: non-negative scalar weighting the critique penalty.
        margin: non-negative scalar preference margin.

    Returns:
        A single float: the mean per-example loss.
    """
    # Your code here
    N = logp_original.shape[0]
    Li = np.zeros((N))
    for i in range(N):
        Li[i] = - logp_revised[i] + lambda_penalty * harm_scores[i] * max(0, logp_original[i] - logp_revised[i] + margin)
    return np.mean(Li)
