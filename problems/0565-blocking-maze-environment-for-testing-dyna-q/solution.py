import numpy as np

def run_blocking_maze(grid_h, grid_w, walls_phase1, walls_phase2,
                      change_step, start, goal, gamma, alpha, epsilon,
                      n_planning, kappa, num_steps, seed=42):
    """
    Simulate a Dyna-Q+ agent in a blocking maze environment.

    Args:
        grid_h: int, grid height
        grid_w: int, grid width
        walls_phase1: list of [row, col] walls before change
        walls_phase2: list of [row, col] walls after change
        change_step: int, timestep at which walls change
        start: [row, col] start position
        goal: [row, col] goal position
        gamma: float, discount factor
        alpha: float, learning rate
        epsilon: float, exploration rate
        n_planning: int, number of planning steps
        kappa: float, exploration bonus coefficient
        num_steps: int, total simulation timesteps
        seed: int, random seed

    Returns:
        Tuple of (cumulative_reward, episodes_completed)
    """
    np.random.seed(seed)

    # ── Convert positions and walls to canonical forms ──────────────────
    start = tuple(start)
    goal = tuple(goal)
    walls1 = set(tuple(w) for w in walls_phase1)
    walls2 = set(tuple(w) for w in walls_phase2)

    # Action deltas: up(0), down(1), left(2), right(3)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ── Agent state ─────────────────────────────────────────────────────
    Q = np.zeros((grid_h, grid_w, 4))          # Q-table
    model = {}                                   # (s, a) -> (reward, next_state)
    last_visit = {}                              # (s, a) -> most recent timestep
    visited_pairs = []                           # ordered list for random sampling

    cumulative_reward = 0.0
    episodes_completed = 0
    state = start                                # agent starts at 'start'

    # ── Helper: environment step ────────────────────────────────────────
    def take_step(s, a, walls):
        """Return (next_state, reward, done) after taking action a in state s."""
        dr, dc = deltas[a]
        nr, nc = s[0] + dr, s[1] + dc
        # Wall or out-of-bounds → stay in place
        if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w or (nr, nc) in walls:
            next_s = s
        else:
            next_s = (nr, nc)
        # Reward & episode termination
        if next_s == goal:
            return next_s, 1.0, True
        return next_s, 0.0, False

    # ── Main loop ───────────────────────────────────────────────────────
    for t in range(1, num_steps + 1):            # timesteps are 1-indexed

        # Determine current wall configuration
        walls = walls2 if t >= change_step else walls1

        # ── Action selection: epsilon-greedy with random tie-breaking ──
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            q_vals = Q[state[0], state[1], :]
            max_q = q_vals.max()
            best_actions = np.where(q_vals == max_q)[0]
            action = int(np.random.choice(best_actions))

        # ── Real environment step ──────────────────────────────────────
        next_state, reward, done = take_step(state, action, walls)

        cumulative_reward += reward

        # ── Q-learning update (real experience) ────────────────────────
        s_idx = (state[0], state[1])
        ns_idx = (next_state[0], next_state[1])
        max_next_q = Q[ns_idx[0], ns_idx[1], :].max()
        Q[s_idx[0], s_idx[1], action] += alpha * (
            reward + gamma * max_next_q - Q[s_idx[0], s_idx[1], action]
        )

        # ── Update model and visit tracking ─────────────────────────────
        sa = (state, action)
        if sa not in model:
            model[sa] = (reward, next_state)
            visited_pairs.append(sa)
        else:
            model[sa] = (reward, next_state)
        last_visit[sa] = t

        # ── Planning updates ───────────────────────────────────────────
        if visited_pairs:
            n = len(visited_pairs)
            indices = np.random.randint(0, n, size=n_planning)
            for idx in indices:
                ps, pa = visited_pairs[idx]
                pr, pns = model[(ps, pa)]
                tau = t - last_visit[(ps, pa)]
                bonus = kappa * np.sqrt(tau)
                # Q-learning-style update with exploration bonus
                pns_idx = (pns[0], pns[1])
                max_pns_q = Q[pns_idx[0], pns_idx[1], :].max()
                Q[ps[0], ps[1], pa] += alpha * (
                    pr + bonus + gamma * max_pns_q - Q[ps[0], ps[1], pa]
                )

        # ── Episode reset ──────────────────────────────────────────────
        if done:
            episodes_completed += 1
            state = start
        else:
            state = next_state

    return cumulative_reward, episodes_completed