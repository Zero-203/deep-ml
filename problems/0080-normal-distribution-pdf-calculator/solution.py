import torch

def normal_pdf(x: torch.Tensor, mean: torch.Tensor, std_dev: torch.Tensor) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated (torch.Tensor scalar).
    :param mean: The mean (mu) of the distribution (torch.Tensor scalar).
    :param std_dev: The standard deviation (sigma) of the distribution (torch.Tensor scalar).
    :return: The PDF value rounded to 5 decimal places.
    """
    # Your code here
    PHI=3.1415926
    return round((torch.exp(-(x-mean)**2/(2*std_dev**2))/(torch.sqrt(torch.as_tensor(2*PHI))*std_dev)).item(),5)