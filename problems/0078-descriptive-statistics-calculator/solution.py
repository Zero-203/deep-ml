import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = torch.as_tensor(data,dtype=torch.float64)
    n = data.shape[0]
    mean = torch.mean(data)
    median = torch.quantile(data, 0.5)
    mode = torch.mode(data)
    var = torch.var(data,correction=0)
    std = torch.std(data,correction=0)
    quantile = torch.quantile(data,torch.tensor([0.25,0.5,0.75],dtype=torch.float64))
    IQR = quantile[2]-quantile[0]
    return {
        "mean":mean.item(),
        "median":median.item(),
        "mode":mode[0].item(),
        "variance":var.item(),
        "standard_deviation":std.item(),
        "25th_percentile":quantile[0].item(),
        "50th_percentile":quantile[1].item(),
        "75th_percentile":quantile[2].item(),
        "interquartile_range":IQR.item()
    }