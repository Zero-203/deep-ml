def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    total_param = 0
    for layer in layers:
        layer_type = layer["type"]
        layer_param = 0
        if layer_type == "dense":
            input_size, output_size, use_bias = layer["input_size"], layer["output_size"], layer["use_bias"]
            layer_param += input_size * output_size
            if use_bias:
                layer_param += output_size
        elif layer_type == "conv2d":
            in_channels, out_channels, kernel_size, use_bias = layer["in_channels"],layer["out_channels"], layer["kernel_size"], layer["use_bias"]
            kernel_param = 0
            if type(kernel_size) == int:
                kernel_param = kernel_size **2
            if type(kernel_size) == tuple:
                w, h = kernel_size
                kernel_param = w * h
            layer_param += in_channels * out_channels * kernel_param
            if use_bias:
                layer_param += out_channels
        elif layer_type == "embedding":
            num_embeddings, embedding_dim = layer["num_embeddings"], layer["embedding_dim"]
            layer_param += num_embeddings * embedding_dim
        else:
            print(f"Error layer type{layer_type}")
        total_param += layer_param
    return total_param

