import torch

def ddim_sample_step(x_t: torch.Tensor, noise_pred: torch.Tensor, alpha_bar_t: float, alpha_bar_t_prev: float) -> tuple:
    """
    Perform one deterministic DDIM sampling step.

    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        noise_pred: Predicted noise from the model, shape (D,)
        alpha_bar_t: Cumulative alpha at timestep t
        alpha_bar_t_prev: Cumulative alpha at timestep t-1

    Returns:
        Tuple of (pred_x0, x_t_prev), each a torch.Tensor rounded to 4 decimals
    """
    D = x_t.shape[0]
    alpha_bar_t, alpha_bar_t_prev = torch.tensor(alpha_bar_t), torch.tensor(alpha_bar_t_prev)
    pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
    x_t_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
    for idx in range(D):
        pred_x0[idx]=round(pred_x0[idx].item(), 4)
        x_t_prev[idx]=round(x_t_prev[idx].item(), 4)
    return pred_x0, x_t_prev


