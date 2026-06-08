import numpy as np

def novel_view_synthesis(source_image: np.ndarray, depth_map: np.ndarray,
                        K_src: np.ndarray, K_tgt: np.ndarray,
                        R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Synthesize a novel view by reprojecting source pixels using depth.

    Args:
        source_image: float array of shape (H, W, C)
        depth_map: float array of shape (H, W), per-pixel depth (>0 valid)
        K_src: 3x3 source camera intrinsic matrix
        K_tgt: 3x3 target camera intrinsic matrix
        R: 3x3 rotation matrix (source to target)
        t: length-3 translation vector (source to target)

    Returns:
        target_image: float array of shape (H, W, C)
    """
    H, W, C = source_image.shape
    Ptgtmat = np.zeros((H,W))
    target_image = np.zeros_like(source_image)
    for h in range(H):
        for w in range(W):
                Ptgtmat[h][w] = 1e9

    for h in range(H):
        for w in range(W):
            psrc = np.array([w,h,1]).reshape(3,-1)
            Psrc = depth_map[h][w] * np.matmul(np.linalg.inv(K_src), psrc)
            Ptgt = np.matmul(R, Psrc) + t.reshape(3, 1)
            proj = np.matmul(K_tgt, Ptgt)
            ptgt = proj/proj[2] if proj[2] !=0 else proj
            h_n, w_n = int(round(ptgt[1][0], 0)),int(round(ptgt[0][0], 0))                
            if 0 <= h_n and h_n < H and 0 <= w_n and w_n <W:
                if proj[2][0] > 0 and Ptgtmat[h_n][w_n] > proj[2][0]:
                    Ptgtmat[h_n][w_n]=proj[2][0]
                    target_image[h_n][w_n] = source_image[h][w]
    return target_image


    