import numpy as np

def tanh_soft_cap(logits, softcap):
    """Apply tanh soft-capping to logits.

    Args:
        logits: numpy array of any shape.
        softcap: positive float, or None/<=0 to disable.

    Returns:
        numpy array of the same shape with soft-capped values,
        rounded to 6 decimal places.
    """
    if softcap is not None and softcap > 0:
        logits = softcap * np.tanh(logits / softcap)
    shape = logits.shape
    logits = np.round(logits, 6)
    return logits
        
