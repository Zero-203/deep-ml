import numpy as np

def reconstruct_teacher_logits(hidden_states, teacher_ids, teacher_heads):
    """
    Reconstruct per-sample teacher logits from cached hidden states.

    Args:
        hidden_states: array-like of shape (N, H)
        teacher_ids: array-like of length N, integer teacher indices
        teacher_heads: list of dicts with 'W' (V, H) and 'b' (V,)

    Returns:
        2D list of shape (N, V) with logits in original sample order.
    """
    # Your code here
    N, H =  len(hidden_states),len(hidden_states[0])
    logits = N * [[]]
    hidden_states = np.array(hidden_states)
    
    for idx in range(N):
        V = len(teacher_heads[0]['b'])
        logits[idx] = np.zeros(V)
        logits[idx] = teacher_heads[teacher_ids[idx]]['W'] @ hidden_states[idx] + teacher_heads[teacher_ids[idx]]['b']
    return logits
