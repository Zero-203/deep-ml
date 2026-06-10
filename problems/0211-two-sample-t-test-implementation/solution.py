import torch
from scipy.stats import t

def two_sample_t_test(sample1, sample2, alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test) using PyTorch.
    
    Args:
        sample1: First sample data (list or torch.Tensor)
        sample2: Second sample data (list or torch.Tensor)
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary containing:
        - t_statistic: The calculated t-statistic
        - p_value: Two-tailed p-value
        - degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
        - reject_null: Boolean, whether to reject null hypothesis
        - cohens_d: Effect size (Cohen's d)
    """
    # Your code here
    sample1_t, sample2_t = torch.as_tensor(sample1), torch.as_tensor(sample2)
    n1, n2 = torch.numel(sample1_t), torch.numel(sample2_t)
    mean_1, mean_2 = torch.mean(sample1_t), torch.mean(sample2_t)
    var_1, var_2 = torch.sum((sample1_t - mean_1)**2)/(n1 - 1), torch.sum((sample2_t - mean_2)**2)/(n2 - 1)
    se = torch.sqrt(var_1 / n1 + var_2 / n2)

    ts = (mean_1 - mean_2)/se
    df = (var_1 / n1 + var_2 / n2)**2 / ((var_1 / n1)**2 / (n1 - 1) + (var_2 / n2)**2 / (n2 - 1))
    p = 2 * (1 - t.cdf(torch.abs(ts).item(),df))
    sp = torch.sqrt(((n1-1)*var_1+(n2-1)*var_2)/(df))
    d = (mean_1 - mean_2)/sp
    reject_null = p < alpha
    
    return {
        't_statistic':ts,
        'p_value':p,
        'degrees_of_freedom':df,
        'reject_null':reject_null,
        'cohens_d':d
    }

    