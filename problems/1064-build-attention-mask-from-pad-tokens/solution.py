import numpy as np

def build_attention_mask(token_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """
    Build a binary attention mask from padded token ids.

    Args:
        token_ids: array of shape (batch_size, seq_len)
        pad_token_id: integer id used for padding

    Returns:
        Integer mask array of shape (batch_size, seq_len) with 1 for real
        tokens and 0 for padding positions.
    """
    batch_size, seq_len = token_ids.shape
    mask = np.ones((batch_size, seq_len), dtype=int)
    for i in range(batch_size):
        for j in range(seq_len):
            mask[i][j] = 0 if token_ids[i][j] == pad_token_id else 1
    return mask 

