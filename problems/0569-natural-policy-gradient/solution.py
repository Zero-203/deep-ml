import numpy as np

def natural_policy_gradient_step(
    theta: np.ndarray,
    trajectories: list,
    feature_vectors: dict,
    actions: list,
    gamma: float,
    alpha: float,
    epsilon: float = 1e-4
) -> dict:
    """
    Perform a single Natural Policy Gradient update step.

    Args:
        theta: (d,) parameter vector for the softmax policy
        trajectories: list of episodes; each episode is a list of (state, action, reward) tuples
        feature_vectors: dict mapping (state, action) -> list of floats (feature vector)
        actions: list of all possible actions
        gamma: discount factor
        alpha: learning rate
        epsilon: regularization constant for FIM inversion

    Returns:
        Dictionary with keys:
          'vanilla_gradient': np.ndarray of shape (d,)
          'fisher_matrix': np.ndarray of shape (d, d)
          'natural_gradient': np.ndarray of shape (d,)
          'updated_theta': np.ndarray of shape (d,)
    """
    d = len(theta)
    total_grad = np.zeros(d)
    total_fisher = np.zeros((d, d))
    total_steps = 0
    num_episodes = len(trajectories)

    # Pre-convert feature vectors to numpy arrays for faster access
    feat_cache = {}
    for key, feat in feature_vectors.items():
        feat_cache[key] = np.asarray(feat, dtype=float)

    for episode in trajectories:
        T = len(episode)
        # Compute discounted returns for this episode
        returns = np.zeros(T)
        G = 0.0
        # Compute from the end
        for t in reversed(range(T)):
            _, _, r = episode[t]
            G = r + gamma * G
            returns[t] = G

        # Accumulate gradient for this episode
        episode_grad = np.zeros(d)
        for t in range(T):
            state, action, _ = episode[t]
            G_t = returns[t]

            # Get feature vectors for all actions in this state
            phi_all = {}
            for a in actions:
                phi_all[a] = feat_cache[(state, a)]

            # Compute scores and softmax probabilities
            scores = {}
            for a in actions:
                scores[a] = np.dot(phi_all[a], theta)
            # Numerical stability: subtract max score
            max_score = max(scores.values())
            exp_scores = {a: np.exp(scores[a] - max_score) for a in actions}
            Z = sum(exp_scores.values())
            probs = {a: exp_scores[a] / Z for a in actions}

            # Expected feature under current policy
            expected_phi = sum(probs[a] * phi_all[a] for a in actions)

            # Score function: φ(s,a) - E[φ(s,·)]
            score = phi_all[action] - expected_phi

            # Accumulate vanilla policy gradient (weighted by discounted return)
            episode_grad += G_t * score

            # Accumulate Fisher information matrix sample
            total_fisher += np.outer(score, score)
            total_steps += 1

        total_grad += episode_grad

    # Average gradient over episodes
    vanilla_gradient = total_grad / num_episodes

    # Average Fisher matrix over all time steps, then regularize
    fisher_matrix = total_fisher / total_steps

    # Compute natural gradient by solving linear system
    natural_gradient = np.linalg.solve(fisher_matrix + epsilon * np.eye(d), vanilla_gradient)

    # Update parameters
    updated_theta = theta + alpha * natural_gradient

    return {
        'vanilla_gradient': vanilla_gradient,
        'fisher_matrix': fisher_matrix,
        'natural_gradient': natural_gradient,
        'updated_theta': updated_theta
    }