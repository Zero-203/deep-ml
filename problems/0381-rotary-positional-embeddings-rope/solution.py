import torch
import numpy as np

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, d = x.shape
    vec_theta = torch.zeros((d>>1,1))
    for i in range(d>>1):
        vec_theta[i]=1/(base**(2*i/d))
    
    for i in range(seq_len):
        sin_vec=torch.sin(positions[i]*vec_theta)
        cos_vec=torch.cos(positions[i]*vec_theta)
        even_pos=x[i][0:d:2].reshape(-1,1)
        odd_pos=x[i][1:d+1:2].reshape(-1,1)
        # print(even_pos,cos_vec)
        new_even_pos=torch.mul(even_pos,cos_vec)-torch.mul(odd_pos,sin_vec)
        # print(new_even_pos)
        new_odd_pos=torch.mul(even_pos,sin_vec)+torch.mul(odd_pos,cos_vec)
        for j in range(d>>1):
            x[i][2*j]=new_even_pos[j]
            x[i][2*j+1]=new_odd_pos[j]

    return x