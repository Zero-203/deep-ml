import torch

def mutual_information(joint_prob: torch.Tensor) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y) as a torch.Tensor
    
    Returns:
        Mutual information I(X;Y)
    """
    # Your code here
    P_x, P_y = torch.sum(joint_prob,dim=1,keepdim=True), torch.sum(joint_prob,dim=0,keepdim=True)
    epsilon = 1e-10
    return (torch.sum(joint_prob*(torch.log(joint_prob+epsilon)-torch.log(P_x+epsilon)-torch.log(P_y+epsilon)))).item()