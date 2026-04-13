import torch

def local_response_normalization(x: torch.Tensor, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> torch.Tensor:
    """Apply Local Response Normalization across channels (similar to AlexNet)."""
    # Your code here
    in_B,in_C,in_H,in_W=x.shape
    y=torch.zeros_like(x)
    
    for i in range(0,in_C):
        sub_mat=x[-1,max(0,i-n/2):min(in_C-1,i+n/2),-1,-1]
        y[0:in_B,i,0:in_H,0:in_W]=x[0:in_B,i,0:in_H,0:in_W]/(k+alpha*sum(sub_mat*sub_mat))**beta

    return y
