import numpy as np

def async_value_iteration(num_states: int, transitions: list, gamma: float, update_order: list) -> list:
    """
    Perform asynchronous value iteration on an MDP.
    
    Args:
        num_states: Number of states in the MDP
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        update_order: Sequence of state indices to update (in order)
    
    Returns:
        List of float values representing the value function
    """
    # Your code here
    vals=[0.0]*num_states
    for pos in update_order:
        transset = transitions[pos]
        for action in transset:
            expect_val=0.0
            for result in action:
                prob,next_state,reward = result
                expect_val+=prob*(reward+gamma*vals[next_state])
            vals[pos]=max(vals[pos],expect_val)
    
    return vals
        
