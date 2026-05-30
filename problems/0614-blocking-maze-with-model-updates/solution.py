import numpy as np

def blocking_maze_dyna_q(rows: int, cols: int, initial_walls: list, added_walls: list,
                         start: list, goal: list, change_step: int,
                         gamma: float, alpha: float, epsilon: float,
                         n_planning: int, n_steps: int, seed: int) -> dict:
    """
    Simulate a Dyna-Q agent in a blocking maze where walls change mid-simulation.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        initial_walls: List of [row, col] wall positions at start
        added_walls: List of [row, col] walls added at change_step
        start: [row, col] start position
        goal: [row, col] goal position
        change_step: Timestep at which added_walls appear
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration probability
        n_planning: Number of planning updates per real step
        n_steps: Total number of simulation steps
        seed: Random seed
    
    Returns:
        Dictionary with episodes_completed, cumulative_reward, 
        model_entries, and stale_entries
    """
    np.random.seed(seed)
    
    # Convert coordinates to state indices
    def state_idx(r, c):
        return r * cols + c
    
    start_idx = state_idx(start[0], start[1])
    goal_idx = state_idx(goal[0], goal[1])
    
    # Precompute wall sets as indices
    initial_walls_set = set(state_idx(r, c) for r, c in initial_walls)
    added_walls_set = set(state_idx(r, c) for r, c in added_walls)
    all_walls_set = initial_walls_set | added_walls_set
    
    # Q-table and model
    n_states = rows * cols
    Q = np.zeros((n_states, 4))
    model = {}  # key: (s, a), value: (next_s, reward)
    
    # Helper to get true next state and reward given current walls
    def get_transition(s, a, walls_set):
        r = s // cols
        c = s % cols
        if a == 0:    # up
            nr, nc = r - 1, c
        elif a == 1:  # down
            nr, nc = r + 1, c
        elif a == 2:  # left
            nr, nc = r, c - 1
        else:         # right
            nr, nc = r, c + 1
        
        # Check boundaries and walls
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or state_idx(nr, nc) in walls_set:
            next_s = s
        else:
            next_s = state_idx(nr, nc)
        reward = 1.0 if next_s == goal_idx else 0.0
        return next_s, reward
    
    state = start_idx
    episodes_completed = 0
    cumulative_reward = 0.0
    
    for t in range(n_steps):
        # Determine current wall set
        walls_set = all_walls_set if t >= change_step else initial_walls_set
        
        # Action selection (epsilon-greedy with np.argmax for ties)
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])
        
        # Take action in real environment
        next_state, reward = get_transition(state, action, walls_set)
        cumulative_reward += reward
        
        # Q-learning update from real experience
        best_next = np.max(Q[next_state])
        td_target = reward + gamma * best_next
        Q[state, action] += alpha * (td_target - Q[state, action])
        
        # Update model (overwrite)
        model[(state, action)] = (next_state, reward)
        
        # Check goal and reset
        if next_state == goal_idx:
            episodes_completed += 1
            state = start_idx
        else:
            state = next_state
        
        # Planning steps (Dyna-Q)
        for _ in range(n_planning):
            if not model:
                break
            # Sample uniformly from observed state-action pairs
            s_a_list = list(model.keys())
            rand_idx = np.random.randint(len(s_a_list))
            s_plan, a_plan = s_a_list[rand_idx]
            s_next_plan, r_plan = model[(s_plan, a_plan)]
            
            # Q-learning update using model's prediction
            best_next_plan = np.max(Q[s_next_plan])
            td_target_plan = r_plan + gamma * best_next_plan
            Q[s_plan, a_plan] += alpha * (td_target_plan - Q[s_plan, a_plan])
    
    # Compute model statistics
    model_entries = len(model)
    stale_entries = 0
    
    # After simulation, the environment contains all walls
    final_walls_set = all_walls_set
    for (s, a), (s_next_stored, r_stored) in model.items():
        true_next, true_r = get_transition(s, a, final_walls_set)
        if true_next != s_next_stored or abs(true_r - r_stored) > 1e-12:
            stale_entries += 1
    
    return {
        "episodes_completed": episodes_completed,
        "cumulative_reward": round(cumulative_reward, 4),
        "model_entries": model_entries,
        "stale_entries": stale_entries
    }