import numpy as np

def compute_perceptual_distance(features_ref, features_gen, channel_weights):
    """
    Compute perceptual distance between two images based on multi-layer deep features.
    
    Args:
        features_ref: list of numpy arrays, each of shape (C_l, H_l, W_l),
                     feature maps from reference image at different network layers
        features_gen: list of numpy arrays with matching shapes,
                     feature maps from generated image
        channel_weights: list of numpy arrays, each of shape (C_l,),
                        learned per-channel importance weights
    
    Returns:
        float: perceptual distance score (0 = perceptually identical)
    """
    D = 0.0
    eps = 1e-9
    for layer in range(len(features_ref)):
        ref_layer = features_ref[layer]
        gen_layer = features_gen[layer]
        C_l, H_l, W_l = ref_layer.shape
        
        # 逐通道 L2 归一化（沿空间维度求和）
        ref_norm = ref_layer / np.sqrt(np.sum(ref_layer ** 2, axis=0, keepdims=True) + eps)
        gen_norm = gen_layer / np.sqrt(np.sum(gen_layer ** 2, axis=0, keepdims=True) + eps)
        
        # 计算加权平方差，并在通道维求和，保留空间维度
        w = channel_weights[layer].reshape(C_l, 1, 1)
        d_l = np.sum(w * (ref_norm - gen_norm) ** 2, axis=0, keepdims=True)  # shape (1, H_l, W_l)
        
        # 求空间平均，累加到总距离
        D_l = np.sum(d_l) / (H_l * W_l)
        D += D_l
        
    return D
