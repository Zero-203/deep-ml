import torch

def expected_action_value(state, act, P, R, V, gamma) -> float:
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: torch.Tensor, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    # Your code here
    e_val=0.0
    for ps in P[state][act]:
        # print(P[state][act][ps],R[state][act][ps],V[ps])
        e_val+=P[state][act][ps]*(R[state][act][ps]+gamma*V[ps].item())
    return e_val
