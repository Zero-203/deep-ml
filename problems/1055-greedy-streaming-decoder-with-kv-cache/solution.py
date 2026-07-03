import numpy as np

def greedy_stream(model, prompt_ids, max_new_tokens: int, eos_id: int) -> list:
    """
    Greedy streaming decoder backed by a KV cache.

    Args:
        model: object exposing prefill(prompt_ids) -> np.ndarray of logits
               and decode_step(token_id) -> np.ndarray of logits.
        prompt_ids: iterable of int token ids.
        max_new_tokens: maximum number of tokens to generate.
        eos_id: token id that terminates generation.

    Returns:
        List of generated token ids (excluding the prompt and EOS).
    """
    res_list = []
    token_cnt = 0

    new_token_id = np.argmax(model.prefill(prompt_ids))
    while(token_cnt < max_new_tokens):
        if new_token_id == eos_id:
            return res_list
        res_list.append(new_token_id.item())
        token_cnt += 1
        new_token_id = np.argmax(model.decode_step(new_token_id))
    return res_list

