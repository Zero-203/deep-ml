import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    adjusted_alpha = round(base_alpha / n_tilings, 4)

    prediction_before = 0.0
    for act_tile in active_tiles:
        prediction_before += weights[act_tile]
    prediction_before = round(prediction_before, 4)

    error = target - prediction_before
    for act_tile in active_tiles:
        weights[act_tile] += adjusted_alpha * error
        weights[act_tile] = round(weights[act_tile],4)
    
    prediction_after = 0.0
    for act_tile in active_tiles:
        prediction_after += weights[act_tile]
    prediction_after = round(prediction_after, 4)

    return {
        'adjusted_alpha':adjusted_alpha,
        'prediction_before':prediction_before,
        'prediction_after':prediction_after,
        'updated_weights':weights
    }

    