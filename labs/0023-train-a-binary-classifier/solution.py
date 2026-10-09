import numpy as np

# 超参数（与原代码保持一致）
hidden_1, hidden_2 = 128, 32
epochs = 50
lr = 0.001
np.random.seed(42)

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    n_samples, n_features = X_train.shape
    
    # 将标签 reshape 成列向量，便于广播
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # 权重初始化（He 初始化，适用于 ReLU）
    W1 = np.random.randn(n_features, hidden_1) * np.sqrt(2.0 / n_features)
    b1 = np.zeros((1, hidden_1))
    W2 = np.random.randn(hidden_1, hidden_2) * np.sqrt(2.0 / hidden_1)
    b2 = np.zeros((1, hidden_2))
    W3 = np.random.randn(hidden_2, 1) * np.sqrt(2.0 / hidden_2)
    b3 = np.zeros((1, 1))
    
    # 激活函数
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        # 裁剪输入防止 exp 溢出
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        Z1 = X_train @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)
        Z3 = A2 @ W3 + b3
        A3 = sigmoid(Z3)  # 输出概率
        
        # 计算二元交叉熵损失（均值）
        epsilon = 1e-15
        A3_clipped = np.clip(A3, epsilon, 1 - epsilon)
        loss = -np.mean(y_train * np.log(A3_clipped) + (1 - y_train) * np.log(1 - A3_clipped))
        
        # 反向传播
        m = n_samples
        dZ3 = A3 - y_train                     # (m, 1)
        dW3 = A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * (Z2 > 0)                   # ReLU 导数
        dW2 = A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X_train.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # SGD 参数更新
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    
    # 返回预测函数
    def predict(X):
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)
        Z3 = A2 @ W3 + b3
        A3 = sigmoid(Z3)
        return (A3 >= 0.5).astype(int).flatten()
    
    return predict