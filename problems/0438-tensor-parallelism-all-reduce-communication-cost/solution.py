def tensor_parallel_allreduce_cost(
    hidden_size: int,
    sequence_length: int,
    batch_size: int,
    num_gpus: int,
    num_layers: int,
    bytes_per_element: int = 2,
    bandwidth_gb_per_sec: float = 300.0,
    allreduces_per_layer: int = 2
) -> dict:
    """
    Calculate the all-reduce communication cost for tensor parallelism.

    Args:
        hidden_size: Hidden dimension of the model
        sequence_length: Sequence length
        batch_size: Micro-batch size per GPU
        num_gpus: Number of GPUs in tensor parallel group
        num_layers: Number of transformer layers
        bytes_per_element: Bytes per element (2 for FP16, 4 for FP32)
        bandwidth_gb_per_sec: Interconnect bandwidth in GB/s
        allreduces_per_layer: Number of all-reduce ops per layer (forward pass)

    Returns:
        Dictionary with communication cost analysis
    """
    message_size_bytes = hidden_size * sequence_length * batch_size * bytes_per_element
    comm_volume_per_allreduce_bytes = 2 * (num_gpus - 1) * message_size_bytes / num_gpus
    total_comm_volume_bytes = num_layers * allreduces_per_layer * comm_volume_per_allreduce_bytes
    total_comm_time_ms = total_comm_volume_bytes / (bandwidth_gb_per_sec* 1e6)
    return {
        "message_size_bytes":message_size_bytes,
        "comm_volume_per_allreduce_bytes":round(comm_volume_per_allreduce_bytes,4),
        "total_comm_volume_bytes":round(total_comm_volume_bytes,4),
        "total_comm_time_ms":round(total_comm_time_ms,4)
    }
    