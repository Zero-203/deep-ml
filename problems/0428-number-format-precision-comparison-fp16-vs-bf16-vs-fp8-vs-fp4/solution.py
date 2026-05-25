import torch
import numpy as np

def compare_formats(values) -> dict:
    """
    Simulate quantizing FP32 values to FP16, BF16, FP8_E4M3, and FP4_E2M1 formats
    using PyTorch's native dtype casting and tensor operations.

    Args:
        values: list, numpy array, or torch.Tensor of float values

    Returns:
        dict mapping format name to dict with keys:
            'max_representable', 'min_positive_normal', 'quantized',
            'max_abs_error', 'mean_abs_error'
    """
    # Convert input to a float32 tensor
    t = torch.as_tensor(values, dtype=torch.float32)

    # ---------- FP16 and BF16 via native casting ----------
    t_fp16 = t.to(torch.float16).to(torch.float32)
    info_fp16 = torch.finfo(torch.float16)
    max_fp16 = info_fp16.max
    min_norm_fp16 = info_fp16.tiny

    t_bf16 = t.to(torch.bfloat16).to(torch.float32)
    info_bf16 = torch.finfo(torch.bfloat16)
    max_bf16 = info_bf16.max
    min_norm_bf16 = info_bf16.tiny

    # ---------- Custom quantizer for FP8 and FP4 ----------
    def quantize_custom(x, exp_bits, man_bits, bias, max_exp, max_finite, has_inf):
        """
        x: float32 tensor
        Returns quantized float32 tensor
        """
        device = x.device
        sign = torch.sign(x)
        abs_x = torch.abs(x)

        zero_mask = abs_x == 0.0
        nan_mask = torch.isnan(x)
        inf_mask = torch.isinf(x)

        quantized = torch.zeros_like(x)

        # NaN -> NaN
        quantized[nan_mask] = float('nan')

        # Inf handling
        if has_inf:
            quantized[inf_mask] = x[inf_mask]   # keep signed inf
        else:
            # Clamp to max_finite with sign
            quantized[inf_mask] = torch.copysign(
                torch.tensor(max_finite, device=device, dtype=x.dtype),
                x[inf_mask]
            )

        # Finite, non-zero values
        finite_mask = ~(zero_mask | nan_mask | inf_mask)
        if not finite_mask.any():
            return quantized

        x_f = x[finite_mask]
        abs_f = abs_x[finite_mask]

        # frexp gives mantissa m in [0.5, 1) and exponent e such that x = m * 2^e
        m_frexp, exp_f = torch.frexp(abs_f)
        e_stored = exp_f + (bias - 1)   # biased exponent for normal numbers

        # Separate ranges
        overflow_mask = e_stored > max_exp
        sub_mask = (e_stored <= 0) & ~overflow_mask
        norm_mask = (e_stored >= 1) & (e_stored <= max_exp) & ~overflow_mask

        result = torch.empty_like(abs_f)

        # ---- Overflow: clamp to max_finite ----
        if overflow_mask.any():
            result[overflow_mask] = max_finite

        # ---- Subnormals ----
        if sub_mask.any():
            abs_sub = abs_f[sub_mask]
            scale = 2.0 ** (bias + man_bits - 1)
            man_int = torch.round(abs_sub * scale).to(torch.int64)
            two_man = 1 << man_bits
            normal_overflow = man_int >= two_man
            val_sub = man_int.float() * (2.0 ** (1 - bias - man_bits))
            min_norm = 2.0 ** (1 - bias)
            val_sub = torch.where(normal_overflow, min_norm, val_sub)
            result[sub_mask] = val_sub

        # ---- Normals ----
        if norm_mask.any():
            abs_norm = abs_f[norm_mask]
            e_norm = e_stored[norm_mask]
            scale = 2.0 ** (e_norm - bias)
            norm_val = abs_norm / scale
            man_float = (norm_val - 1.0) * (2 ** man_bits)
            man_int = torch.round(man_float).to(torch.int32)
            two_man = 1 << man_bits
            carry = (man_int == two_man)
            e_adj = e_norm + carry.int()
            man_adj = torch.where(carry, 0, man_int)

            # Post‑rounding overflow / NaN pattern
            overflow_after = e_adj > max_exp
            all_ones = (1 << man_bits) - 1
            nan_pattern = (e_adj == max_exp) & (man_adj == all_ones)

            value = (1.0 + man_adj.float() / two_man) * (2.0 ** (e_adj.float() - bias))
            clamp_mask = overflow_after | nan_pattern
            value = torch.where(
                clamp_mask,
                torch.tensor(max_finite, device=device, dtype=value.dtype),
                value
            )
            result[norm_mask] = value

        # Apply sign and store
        quantized[finite_mask] = torch.copysign(result, x_f)
        return quantized

    # FP8_E4M3 parameters
    exp8, man8 = 4, 3
    bias8 = 7
    max_exp8 = 15
    max_finite8 = (2 - 2 ** (1 - man8)) * 2 ** (max_exp8 - bias8)   # = 448
    min_norm8 = 2.0 ** (1 - bias8)   # 0.015625

    t_fp8 = quantize_custom(t, exp8, man8, bias8, max_exp8, max_finite8, has_inf=False)

    # FP4_E2M1 parameters
    exp4, man4 = 2, 1
    bias4 = 1
    max_exp4 = 3
    max_finite4 = (2 - 2 ** (1 - man4)) * 2 ** (max_exp4 - bias4)   # = 4
    min_norm4 = 2.0 ** (1 - bias4)   # 1.0

    t_fp4 = quantize_custom(t, exp4, man4, bias4, max_exp4, max_finite4, has_inf=False)

    # ---------- Error computation ----------
    def compute_errors(orig, quant):
        finite_orig = torch.isfinite(orig)
        inf_quant = torch.isinf(quant)
        mask_inf_err = inf_quant & finite_orig
        errors = torch.where(
            mask_inf_err,
            torch.tensor(float('inf'), device=orig.device),
            (orig - quant).abs()
        )
        max_err = torch.max(errors).item()
        mean_err = torch.mean(errors.float()).item()
        return max_err, mean_err

    # ---------- Rounding helper ----------
    def round_list(tensor):
        return [
            round(v.item(), 7) if torch.isfinite(v) else v
            for v in tensor
        ]

    # ---------- Build result dict ----------
    results = {}
    for name, qt, max_val, min_val in [
        ('fp16', t_fp16, max_fp16, min_norm_fp16),
        ('bf16', t_bf16, max_bf16, min_norm_bf16),
        ('fp8_e4m3', t_fp8, max_finite8, min_norm8),
        ('fp4_e2m1', t_fp4, max_finite4, min_norm4)
    ]:
        max_err, mean_err = compute_errors(t, qt)
        results[name] = {
            'max_representable': round(float(max_val), 6),
            'min_positive_normal': float(min_val),
            'quantized': round_list(qt),
            'max_abs_error': round(max_err, 6) if max_err != float('inf') else float('inf'),
            'mean_abs_error': round(mean_err, 6) if mean_err != float('inf') else float('inf')
        }

    return results