import torch
from scipy.stats import t

def confidence_interval(data: torch.Tensor, confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean.
    
    Args:
        data: Sample data as torch.Tensor
        confidence_level: Confidence level (default 0.95)
    
    Returns:
        Dictionary containing:
        - mean: Sample mean (point estimate)
        - standard_error: Standard error of the mean
        - margin_of_error: Margin of error
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - confidence_level: Confidence level used
    """
    # Your code here
    n = torch.numel(data)
    mean = torch.mean(data)
    standard_error = torch.sqrt(torch.mean((data-mean)**2)/(n-1))
    degrees_freedom = n - 1
    alpha = 1 - confidence_level
    critical_value = t.ppf(1 - alpha / 2, df=degrees_freedom)
    margin_of_error = critical_value * standard_error
    return {
        'mean':mean,
        'standard_error':standard_error,
        'margin_of_error':margin_of_error,
        'lower_bound':mean-margin_of_error,
        'upper_bound':mean+margin_of_error,
        'confidence_level':confidence_level
    }

    