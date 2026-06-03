def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    sum_tokens, sum_elapse = 0,0.0
    step_tps, cumulative_tps, overall_tps = [], [], 0.0
    for tokens_processed, elapsed_seconds in intervals:
        if tokens_processed < 0:
            raise ValueError("Negative processed tokens.")
        if elapsed_seconds <= 0:
            raise ValueError("Negative elapsed seconds.")
        step_tps.append(round(tokens_processed/elapsed_seconds,2))
        sum_tokens += tokens_processed
        sum_elapse += elapsed_seconds
        cumulative_tps.append(round(sum_tokens/sum_elapse,2))
        overall_tps = sum_tokens/sum_elapse
    return {
        'step_tps':step_tps,
        'cumulative_tps':cumulative_tps,
        'overall_tps':round(overall_tps,2)
    }

