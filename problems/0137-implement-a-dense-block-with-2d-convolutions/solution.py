import torch
import torch.nn.functional as F

def dense_net_block(input_data: torch.Tensor, num_layers: int, growth_rate: int, kernels: list, kernel_size: tuple = (3, 3)) -> torch.Tensor:
    kh, kw = kernel_size
    padding = (kh - 1) // 2
    
    # Input is NHWC, convert to NCHW for PyTorch operations
    concatenated_features = input_data.permute(0, 3, 1, 2).clone()  # NHWC -> NCHW
    
    for l in range(num_layers):
        # Check channel dimension matches kernel's expected input channels
        current_channels = concatenated_features.shape[1]
        kernel = kernels[l]  # Shape: (kh, kw, in_channels, out_channels)
        expected_in_channels = kernel.shape[2]
        
        if current_channels != expected_in_channels:
            raise ValueError(f"Channel mismatch: kernel expects {expected_in_channels}, got {current_channels}")
        
        # ReLU activation using PyTorch built-in
        activated = torch.relu(concatenated_features)
        
        # Convert kernel from NHWC format (kh, kw, in_channels, out_channels) 
        # to PyTorch format (out_channels, in_channels, kh, kw)
        kernel_nchw = kernel.permute(3, 2, 0, 1)
        
        # Convolution using PyTorch's built-in F.conv2d with same padding
        conv_output = F.conv2d(activated, kernel_nchw, padding=padding)
        
        # Concatenate along channel axis (dim=1 in NCHW format)
        concatenated_features = torch.cat([concatenated_features, conv_output], dim=1)
    
    # Convert back to NHWC format
    return concatenated_features.permute(0, 2, 3, 1)  # NCHW -> NHWC