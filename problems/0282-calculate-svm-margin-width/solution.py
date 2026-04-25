import torch

def svm_margin_width(w: torch.Tensor) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : torch.Tensor - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    w_l2=torch.sqrt(torch.sum(w**2))
    return 2/w_l2.item()