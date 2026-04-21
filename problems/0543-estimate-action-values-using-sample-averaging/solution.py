def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.
    
    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step
        
    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    Q=[0]*k
    N=[0]*k
    for i in range(len(actions)):
        N[actions[i]]+=1
        Q[actions[i]]+=rewards[i]

    for i in range(k):
        Q[i]=round(Q[i]/N[i] if N[i] != 0 else 0.0, 4)
    
    return Q,N