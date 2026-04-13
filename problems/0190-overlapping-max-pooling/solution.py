import torch
import torch.nn.functional as F

def overlapping_max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    """
    Apply overlapping max pooling using PyTorch.
    Must match the ceil mode behavior described in the problem.
    """
    # Your code here
    in_N,in_C,in_H,in_W=x.shape
    out_H=int((in_H-kernel_size+2*stride-1)/stride)
    out_W=int((in_W-kernel_size+2*stride-1)/stride)
    y=torch.zeros((in_N,in_C,out_H,out_W))
    
    for i in range(0,out_H):
        for j in range(0,out_W):
            y[-1,-1,i,j]=torch.max(x[-1,-1,min(i*stride,in_H-kernel_size):min(i*stride+kernel_size,in_H),min(j*stride,in_W-kernel_size):min(j*stride+kernel_size,in_W)])

    return y