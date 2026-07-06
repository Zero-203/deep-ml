import numpy as np

def expected_vs_sample_updates(P: np.ndarray, R: np.ndarray, gamma: float, alpha: float, num_sweeps: int, seed: int = 42) -> tuple:
    """
    Compare expected and sample updates for Q-value estimation.
    
    Args:
        P: np.ndarray of shape (num_states, num_actions, num_states), transition probabilities
        R: np.ndarray of shape (num_states, num_actions), rewards
        gamma: float, discount factor
        alpha: float, learning rate for sample updates
        num_sweeps: int, number of sweeps through all state-action pairs
        seed: int, random seed for reproducibility
    
    Returns:
        tuple: (Q_expected, Q_sample) both np.ndarray of shape (num_states, num_actions)
    """
    rng = np.random.RandomState(seed)
    num_states, num_actions, _ = P.shape
    _, num_actions = R.shape
    Q_expected, Q_sample = np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))
    Q_expected_new = np.zeros_like(Q_expected)
    for idx in range(num_sweeps):
        for s in range(num_states):
            for a in range(num_actions):
                Q_expected_new[s][a] = R[s][a] + gamma * np.sum(P[s][a]*np.max(Q_expected, axis=1))
                next_state = rng.choice(num_states, p=P[s, a])
                Q_sample[s][a] = Q_sample[s][a] + alpha * (R[s][a] + gamma * np.max(Q_sample[next_state]) - Q_sample[s][a])
        Q_expected = np.copy(Q_expected_new)

    Q_expected, Q_sample = np.round(Q_expected, 4), np.round(Q_sample, 4)
    return Q_expected, Q_sample