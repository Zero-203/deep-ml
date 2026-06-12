import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    tuple_list = []
    score_n = len(scores)
    for idx, score in enumerate(scores):
        tuple_list.append((score,idx))
    ac_idx = []
    sum_score = 0.0
    ac_n = 0
    limit = score_n if n_select is None else n_select
    tuple_list.sort(key = lambda x:x[0], reverse = True)
    for idx in range(score_n):
        if tuple_list[idx][0] >= threshold:
            ac_n += 1
            if len(ac_idx) < limit:
                ac_idx.append(tuple_list[idx][1])
                sum_score += tuple_list[idx][0]

    return {
        'accepted_indices':ac_idx,
        'acceptance_rate':round(ac_n/score_n if score_n > 0 else 0.0, 4),
        'mean_quality':round(sum_score/len(ac_idx) if len(ac_idx) > 0 else 0.0, 4)
    }