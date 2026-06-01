import torch

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    torch.manual_seed(seed)
    # Your implementation here
    if distribution == "uniform":
        func = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        mu, sig = 0.5, torch.sqrt(torch.tensor(1/12))
    elif distribution == "exponential":
        func = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        mu, sig = 1.0, 1.0
    elif distribution == "bernoulli":
        func = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.3]))
        mu, sig = 0.3, torch.sqrt(torch.tensor(0.21))

    result_means = torch.zeros((runs),dtype=torch.float64)
    result_stds = torch.zeros((runs),dtype=torch.float64)
    temp = torch.zeros((n),dtype=torch.float64)
    for i in range(runs):
        for j in range(n):
            temp[j] = func.sample()
        mean = (torch.mean(temp) - mu) * torch.sqrt(torch.tensor(n)) /sig
        std = torch.sqrt(torch.sum(temp**2)/n-torch.sum(temp/n)**2)
        result_means[i]=mean
        result_stds[i]=std
    
    mean = torch.mean(result_means)
    std = torch.sqrt(torch.sum(result_means**2)/runs-torch.sum(result_means/runs)**2)

    return {
        "mean":mean.item(),
        "std":std.item()
    }
    