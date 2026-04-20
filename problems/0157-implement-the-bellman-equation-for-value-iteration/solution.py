import torch

def bellman_update(V: torch.Tensor, transitions: list, gamma: float) -> torch.Tensor:
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: torch.Tensor, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      torch.Tensor, updated state values
    """
    V_new=torch.zeros_like(V)
    n_states=torch.numel(V)
    for i in range(n_states):
      val_new=0.0
      for start in transitions[i]:
        val_temp=0.0
        for tp in transitions[i][start]:
          val_temp+=tp[0]*(tp[2]+gamma*V[tp[1]])
        val_new=max(val_new,val_temp)
      V_new[i]=val_new
    return V_new