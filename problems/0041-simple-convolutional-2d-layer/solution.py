import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    height,width=input_matrix.shape
    kernel_height,kernel_width=kernel.shape
    out_height=int((2*padding+height-kernel_height+stride)/stride)
    out_width=int((2*padding+width-kernel_width+stride)/stride)
    output_matrix=torch.zeros((out_height,out_width))

    paded_matrix=torch.zeros([2*padding+height,2*padding+width]);
    paded_matrix[padding:padding+height,padding:padding+width]=input_matrix
    input_matrix=paded_matrix

    for i in range(0,out_height):
        for j in range(0,out_width):
            output_matrix[i][j]=sum(sum(kernel*input_matrix[i*stride:i*stride+kernel_height,j*stride:j*stride+kernel_width]))

    return output_matrix