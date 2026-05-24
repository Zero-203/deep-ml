import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x_exp=np.exp(x)
    return x_exp/np.sum(np.exp(x), axis=-1, keepdims=True)

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    eps=1e-5
    B,T,D=x.shape
    H = num_heads
    x_ln1 = params["ln1_gamma"]*(x-np.mean(x,-1, keepdims=True))/np.sqrt(np.var(x,-1, keepdims=True)+eps)+params["ln1_beta"]
    d=int(D/num_heads)
    Q,K,V=x_ln1 @ params["W_q"], x_ln1 @ params["W_k"], x_ln1 @ params["W_v"]
    Q,K,V = Q.reshape(B,H,T,d),K.reshape(B,H,T,d),V.reshape(B,H,T,d),
    x_attn = np.zeros((B,T,D))
    for b in range(B):
        for h in range(H):
            x_sub = softmax(Q[b,h] @ K[b,h].T/np.sqrt(d)) @ V[b,h]
            x_attn[b,:,d*h:d*(h+1)]=x_sub
    x_attn = x_attn @ params["W_o"]
    x = x + x_attn
    x_ln2 = params["ln2_gamma"]*(x-np.mean(x,-1, keepdims=True))/np.sqrt(np.var(x,-1, keepdims=True)+eps)+params["ln2_beta"]
    x_ffn = gelu(x_ln2 @ params["W_ff1"]+params["b_ff1"]) @ params["W_ff2"] + params["b_ff2"]
    return x + x_ffn 

