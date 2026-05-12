import numpy as np
from typing import List, Tuple

def intra_option_q_learning(
    num_states: int,
    num_actions: int,
    num_options: int,
    option_policies: List[List[List[float]]],
    option_terminations: List[List[float]],
    initiation_sets: List[List[int]],
    transitions: List[Tuple[int, int, float, int]],
    alpha: float,
    gamma: float
) -> List[List[float]]:
    """
    Learn option-value function Q(s, o) using intra-option Q-learning.

    Args:
        num_states: Total number of states.
        num_actions: Total number of primitive actions.
        num_options: Total number of options.
        option_policies: option_policies[o][s][a] = P(a | s, o)
        option_terminations: option_terminations[o][s] = P(terminate | s, o)
        initiation_sets: initiation_sets[o] = list of states where option o can start
        transitions: List of (state, action, reward, next_state) tuples
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        2D list of Q-values, shape (num_states, num_options), rounded to 4 decimals.
    """

    # Initialize Q(s, o) to zero for all states and options
    Q = [[0.0 for _ in range(num_options)] for _ in range(num_states)]

    # Convert initiation sets to a list of sets for fast membership testing
    initiation_sets_set = [set(states) for states in initiation_sets]

    for s, a, r, s_next in transitions:
        for o in range(num_options):
            if option_policies[o][s][a] <= 0:
                continue
            # Termination probability of option o in the next state
            beta = option_terminations[o][s_next]

            # Options that can be initiated in s_next
            available_options = [op for op in range(num_options) if s_next in initiation_sets_set[op]]

            # max over Q(s_next, o') for available options, 0 if none are available
            max_Q_next = max((Q[s_next][op] for op in available_options), default=0.0)

            # U(s', o) = (1 - beta) * Q(s', o) + beta * max_{o'} Q(s', o')
            U = (1.0 - beta) * Q[s_next][o] + beta * max_Q_next

            # TD error and update
            delta = r + gamma * U - Q[s][o]
            Q[s][o] += alpha * delta

    # Round to 4 decimal places before returning
    rounded_Q = [[round(Q[s][o], 4) for o in range(num_options)] for s in range(num_states)]
    return rounded_Q