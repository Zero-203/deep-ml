import numpy as np

def train_sae(X, W_enc, W_dec, b_enc, lr: float, l1_coef: float, num_steps: int) -> float:
    """
    Train a ReLU sparse autoencoder with full-batch SGD (NumPy version).
    Returns the final loss as a Python float.
    """
    # 转换为 float64 数组
    X = np.asarray(X, dtype=np.float64)
    W_enc = np.asarray(W_enc, dtype=np.float64)
    W_dec = np.asarray(W_dec, dtype=np.float64)
    b_enc = np.asarray(b_enc, dtype=np.float64).reshape(-1)  # 确保是一维向量 (hidden_dim,)
    
    N = X.shape[0]  # 样本数

    for _ in range(num_steps):
        # ----- 前向传播 -----
        pre = X @ W_enc + b_enc          # (N, hidden_dim)
        z = np.maximum(0, pre)           # ReLU
        x_hat = z @ W_dec                # (N, input_dim)

        # ----- 反向传播（手动计算梯度）-----
        # 重构误差对 x_hat 的梯度
        dx_hat = (2.0 / (N * X.shape[1])) * (x_hat - X)                     # (N, input_dim)
        # 损失对 z 的梯度：重构部分 + 稀疏惩罚部分
        dL_dz = dx_hat @ W_dec.T + (l1_coef / N) * np.ones_like(z)   # (N, hidden_dim)
        # ReLU 的梯度
        dpre = dL_dz * (pre > 0)                             # (N, hidden_dim)

        # 参数梯度
        dW_enc = X.T @ dpre                                  # (input_dim, hidden_dim)
        db_enc = np.sum(dpre, axis=0)                        # (hidden_dim,)
        dW_dec = z.T @ dx_hat                                # (hidden_dim, input_dim)

        # ----- 参数更新（全批量 SGD）-----
        W_enc -= lr * dW_enc
        W_dec -= lr * dW_dec
        b_enc -= lr * db_enc

    # ----- 最终损失计算 -----
    pre = X @ W_enc + b_enc
    z = np.maximum(0, pre)
    x_hat = z @ W_dec
    mse = np.mean((x_hat - X) ** 2)
    sparse_penalty = l1_coef * np.mean(np.sum(z, axis=1))
    final_loss = mse + sparse_penalty

    return float(final_loss)   # 返回普通 Python 浮点数