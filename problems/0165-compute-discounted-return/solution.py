import torch

def discounted_return(rewards, gamma: float) -> float:
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or torch.Tensor): List or tensor of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    # Your code herereward_arr = np.array(rewards)
    rewards_t = torch.as_tensor(rewards)
    total_remain = torch.tensor(0.0)
    cur_gamma = 1.0
    elenum = rewards_t.shape[0]
    for idx in range(elenum):
        total_remain += rewards_t[idx] * cur_gamma
        cur_gamma *= gamma
    return total_remain.item()
    