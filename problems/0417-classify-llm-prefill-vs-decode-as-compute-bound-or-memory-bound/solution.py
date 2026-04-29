import torch

def classify_llm_phases(num_params: torch.Tensor, sequence_length: torch.Tensor, batch_size: torch.Tensor, bytes_per_param: torch.Tensor, peak_flops: torch.Tensor, peak_bandwidth: torch.Tensor) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters (scalar tensor or int)
        sequence_length: Number of input tokens processed during prefill (scalar tensor or int)
        batch_size: Number of sequences processed in parallel during decode (scalar tensor or int)
        bytes_per_param: Memory footprint per parameter, e.g. 2 for FP16 (scalar tensor or int)
        peak_flops: Hardware peak compute throughput in FLOP/s (scalar tensor or float)
        peak_bandwidth: Hardware peak memory bandwidth in bytes/s (scalar tensor or float)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = peak_flops / peak_bandwidth

    pre_flops = 2 * num_params * sequence_length
    pre_mem = num_params * bytes_per_param
    pre_ai = pre_flops / pre_mem
    pre_bottleneck = "memory-bound" if pre_ai < ridge_point else "compute-bound"
    pre_ach_flops = pre_flops / pre_mem * peak_bandwidth\
                    if pre_ai < ridge_point else peak_flops
    pre_util_rate = pre_ach_flops*100/peak_flops

    dec_flops = 2 * num_params * batch_size
    dec_mem = num_params * bytes_per_param
    dec_ai = dec_flops / dec_mem
    dec_bottleneck = "memory-bound" if dec_ai < ridge_point else "compute-bound"
    dec_ach_flops = dec_flops / dec_mem * peak_bandwidth\
                    if dec_ai < ridge_point else peak_flops
    dec_util_rate = dec_ach_flops*100/peak_flops

    return {
        "ridge_point":round(ridge_point.item(),4),
        "prefill":{
            "total_flops":round(pre_flops.item(),4),
            "memory_bytes":round(pre_mem.item(),4),
            "arithmetic_intensity":round(pre_ai.item(), 4),
            "bottleneck":pre_bottleneck,
            "achieved_flops":round(pre_ach_flops.item(), 4),
            "utilization_percent":round(pre_util_rate.item(), 4)
        },
        "decode":{
            "total_flops":round(dec_flops.item(),4),
            "memory_bytes":round(dec_mem.item(),4),
            "arithmetic_intensity":round(dec_ai.item(), 4),
            "bottleneck":dec_bottleneck,
            "achieved_flops":round(dec_ach_flops.item(), 4),
            "utilization_percent":round(dec_util_rate.item(), 4)
        }
    }