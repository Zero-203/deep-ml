import numpy as np

def process_variable_resets(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> dict:
    """
    Process trajectory data from parallel environments with variable reset times.
    
    Args:
        rewards: np.ndarray of shape (T, N), rewards at each timestep per environment
        dones: np.ndarray of shape (T, N), boolean terminal flags
        gamma: float, discount factor
    
    Returns:
        Dictionary with keys:
            'episode_returns': list of floats (completed episode discounted returns)
            'episode_lengths': list of ints (completed episode lengths)
            'partial_returns': list of N floats (ongoing episode returns)
            'partial_lengths': list of N ints (ongoing episode lengths)
            'mean_return': float (mean of completed returns, 0.0 if none)
            'total_episodes': int (number of completed episodes)
    """
    T, N = rewards.shape
    ep_ret, ep_len=[], []
    par_ret,par_len = [0.0] * N,[0] * N
    mean_ret, tol_ep = 0.0, 0

    for t in range(T):
        for n in range(N):
            par_ret[n]+=gamma**par_len[n]*rewards[t][n]
            par_len[n]+=1
            if dones[t][n]:
                ep_ret.append(round(float(par_ret[n]),4))
                ep_len.append(int(par_len[n]))
                par_ret[n]=0.0
                par_len[n]=0
    
    par_ret=list(map(lambda x: round(float(x), 4),par_ret))
    mean_ret = round(float(np.mean(ep_ret)), 4) if ep_ret else 0.0
    tol_ep = len(ep_ret)
    return {
        "episode_returns":ep_ret,
        "episode_lengths":ep_len,
        "partial_returns":par_ret,
        "partial_lengths":par_len,
        "mean_return":mean_ret,
        "total_episodes":tol_ep
    }