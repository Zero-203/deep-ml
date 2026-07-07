import torch

def flow_epe_torch(pred, gt, mask=None, max_flow=None) -> torch.Tensor:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: tensors or array-likes of shape (H, W, 2).
        mask: optional tensor/array broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float to clip per-pixel EPE.

    Returns:
        torch.Tensor scalar. Returns tensor(-1.0) on invalid input or if no valid pixels.
    """
    p = torch.as_tensor(pred, dtype=torch.float32)
    g = torch.as_tensor(gt,   dtype=torch.float32)
    H, W, _ = p.shape

    # Your implementation here
    EPE = torch.sqrt(torch.sum((p-g)**2,dim=2))
    if mask is not None:
        EPE = EPE * mask
    if max_flow is not None:
        EPE = torch.clamp(EPE, max=max_flow)
    EPE_mean = torch.sum(EPE) / ((H * W) if mask is None else torch.sum(mask))
    return EPE_mean