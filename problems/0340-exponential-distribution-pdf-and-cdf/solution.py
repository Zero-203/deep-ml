import torch
import math

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties using PyTorch.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    # Your code here
    if lam <= 0:
        return {'pdf':None,'cdf':None,'mean':None,'variance':None}
    pdf,cdf,mean,variance=[],[],1/lam,1/(lam**2)
    for sample in x:
        spdf = lam * math.exp(-lam*sample) if sample >= 0 else 0.0
        scdf = 1 - math.exp(-lam*sample) if sample >= 0 else 0.0
        pdf.append(round(spdf,4))
        cdf.append(round(scdf,4))
    
    return {
        'pdf':pdf,
        'cdf':cdf,
        'mean':round(mean,4),
        'variance':round(variance,4)
    }