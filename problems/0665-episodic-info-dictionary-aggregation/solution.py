import numpy as np

def aggregate_episodic_info(infos: list) -> dict:
    """
    Aggregate episodic statistics from a list of step-level info dictionaries.
    
    Args:
        infos: List of info dictionaries from environment steps.
               Each dict may contain an 'episode' key with sub-dict
               having 'r' (total reward) and 'l' (length) keys.
    
    Returns:
        Dictionary with aggregated episode statistics.
    """
    num_episodes, mean_reward, mean_length, min_reward, max_reward, min_length, max_length = 0, 0.0, 0.0, 0.0, 0.0, 0, 0
    rewards, lengths = [], []

    for info in infos:
        episode = info.get("episode")
        if episode is None:
            continue
        rewards.append(episode["r"])
        if episode["l"]:
            lengths.append(episode["l"])

    num_episodes = len(rewards)
    mean_reward = np.mean(rewards) if len(rewards) else 0.0
    mean_length = np.mean(lengths) if len(lengths) else 0.0
    min_reward = min(rewards) if len(rewards) else 0.0
    max_reward = max(rewards) if len(rewards) else 0.0
    min_length = min(lengths) if len(lengths) else 0
    max_length = max(lengths) if len(lengths) else 0
    return {
        "num_episodes":num_episodes,
        "mean_reward":mean_reward,
        "mean_length":mean_length,
        "min_reward":min_reward,
        "max_reward":max_reward,
        "min_length":min_length,
        "max_length":max_length
    }
    