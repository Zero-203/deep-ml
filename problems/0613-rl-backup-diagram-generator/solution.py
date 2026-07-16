def rl_backup_diagram(
    transitions: dict,
    policy: list,
    state: int,
    backup_type: str,
    V: list,
    gamma: float
) -> dict:
    """
    Generate a structured backup diagram for RL value estimation.
    
    Args:
        transitions: Dict mapping (state, action) -> list of (prob, next_state, reward)
        policy: 2D list where policy[s][a] = probability of action a in state s
        state: The state to compute the backup for
        backup_type: 'v_expectation' or 'v_optimal'
        V: List of current state value estimates
        gamma: Discount factor
    
    Returns:
        Dictionary with keys: action_values, aggregation, backed_up_value, diagram
    """
    state_cnt, action_cnt = len(policy), len(policy[0])
    action_values = {}
    state_values = {}
    aggregation = ""
    q = {}
    diagram = []
    backed_up_value = 0.0
    
    if backup_type == 'v_expectation':
        aggregation = 'expectation'
    elif backup_type == 'v_optimal':
        aggregation = 'maximization'
    else:
        raise Exception("Unexpected backuptype", backup_type)
    
    for idx_s,idx_a in transitions.keys():
        state_values[idx_s] = 0.0
        action_values[idx_a] = 0.0
        if len(diagram) <= idx_a:
            diagram.append({
                'action':idx_a,
                'policy_prob':0.0,
                'transitions':[],
                'action_value':0.0
                })
        for prob,next_state,reward in transitions[(idx_s,idx_a)]:
            q[(idx_s,idx_a)] = prob * (reward+ gamma * V[next_state])
            action_values[idx_a] += q[(idx_s,idx_a)]
            diagram[idx_a]["policy_prob"]=policy[idx_s][idx_a]
            diagram[idx_a]["transitions"].append({
                'next_state':next_state,
                'prob':prob,
                'reward':reward,
                'target':round(reward + gamma * V[next_state],4),
                'contribution':round(prob * (reward+ gamma * V[next_state]),4)
                })

    for idx_s,idx_a in transitions.keys():
        action_values[idx_a]=round(action_values[idx_a], 4)
        diagram[idx_a]["action_value"]=action_values[idx_a]
        if aggregation == 'expectation':
                state_values[idx_s] += policy[idx_s][idx_a]*action_values[idx_a]
        if aggregation == 'maximization':
                state_values[idx_s] = max(action_values[idx_a],state_values[idx_s])
        state_values[idx_s] = round(state_values[idx_s], 4)

    for _,val in state_values.items():
        backed_up_value += val
            
    return {
        'action_values':action_values,
        'aggregation':aggregation,
        'backed_up_value':backed_up_value,
        'diagram':diagram
    }
