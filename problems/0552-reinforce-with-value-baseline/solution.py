import numpy as np

def reinforce_with_baseline(episode, theta, w, gamma, alpha_theta, alpha_w):
    """
    Perform one episode update of REINFORCE with a value function baseline.
    
    Args:
        episode: list of (state, action, reward) tuples
        theta: np.ndarray of shape (n_states, n_actions), policy parameters
        w: np.ndarray of shape (n_states,), value function parameters
        gamma: float, discount factor
        alpha_theta: float, policy learning rate
        alpha_w: float, value function learning rate
    
    Returns:
        tuple: (theta_new, w_new) updated parameters
    """
    n_states, n_actions = theta.shape
    episode_n = len(episode)
    Gta = np.zeros([episode_n])
    idx = episode_n - 1 
    while 0 <= idx:
        if idx < episode_n - 1: 
            Gta[idx] = gamma * Gta[idx+1] + episode[idx][2]
        else:
            Gta[idx] = episode[idx][2]
        idx -= 1
    t = 0
    for state, action, reward in episode:
        Gt = Gta[t]
        action_max_idx = np.argmax(theta[state])
        pi = np.exp(theta[state]-theta[state][action_max_idx])/np.sum(np.exp(theta[state]-theta[state][action_max_idx]))
        grad_theta_sb = - np.ones([n_actions]) * pi
        grad_theta_sb[action] += 1.0
        v = w[state]
        sigma_t = Gt - v
        w[state] += alpha_w * sigma_t
        theta[state] += alpha_theta * gamma**t * sigma_t * grad_theta_sb
        t += 1
    return theta, w