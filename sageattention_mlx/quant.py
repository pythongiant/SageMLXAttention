"""
Quantization utilities for SageAttention MLX.

Implements INT8 and FP8 quantization schemes optimized for Apple Silicon.
"""

import mlx.core as mx
from typing import Tuple
import numpy as np


def quantize_qk(
    x: mx.array,
    gran: str = "per_block",
    block_size: int = 64,
    sm_scale: float = 1.0,
    is_q: bool = True
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quantize Q or K tensor to INT8 using symmetric quantization (NVIDIA aligned).
    
    Parameters
    ----------
    x : mx.array
        Input tensor to quantize, typically Q or K.
    gran : str
        Quantization granularity: "per_block" or "per_thread".
    block_size : int
        Block size for per_block quantization.
    sm_scale : float
        Scale for softmax (applied before quantization for Q only, like NVIDIA).
    is_q : bool
        Whether quantizing Q (True) or K (False).
        Q includes sm_scale * 1.44269504; K uses sm_scale=1.0.
        
    Returns
    -------
    x_int8 : mx.array
        Quantized tensor (int8)
    scale : mx.array
        Quantization scales (symmetric: max(abs) / 127)
    zero_point : mx.array
        Quantization zero points (zeros for symmetric quantization)
    """
    
    if gran == "per_block":
        return _quantize_per_block(x, block_size, sm_scale=sm_scale, is_q=is_q)
    elif gran == "per_thread":
        return _quantize_per_thread(x, sm_scale=sm_scale, is_q=is_q)
    else:
        raise ValueError(f"Unknown quantization granularity: {gran}")


def _quantize_per_block(
    x: mx.array,
    block_size: int = 64,
    sm_scale: float = 1.0,
    is_q: bool = True
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Per-block INT8 quantization matching NVIDIA SageAttention.
    
    Uses symmetric quantization (max absolute value / 127) like NVIDIA.
    For Q: scales include sm_scale * 1.44269504 (exp2 optimization).
    For K: uses sm_scale=1.0 (unscaled).
    """
    # Expect input shape (B, H, L, D)
    if x.ndim != 4:
        # Fall back to simpler approach for other shapes
        shape = x.shape
        x_flat = mx.reshape(x, (-1, x.shape[-1]))
        n_samples = x_flat.shape[0]
        n_blocks = max(1, n_samples // block_size)
        actual_block_size = max(1, n_samples // n_blocks)
        pad_size = (n_blocks * actual_block_size) - n_samples
        if pad_size > 0:
            x_flat = mx.pad(x_flat, ((0, pad_size), (0, 0)))
        x_blocked = mx.reshape(x_flat, (n_blocks, actual_block_size, -1))
        
        # Symmetric quantization
        x_abs_max = mx.max(mx.abs(x_blocked), axis=(1, 2), keepdims=True)
        scales = x_abs_max / 127.0
        scales = mx.where(scales == 0, mx.ones_like(scales) * 1e-6, scales)
        x_int8 = mx.round(x_blocked / scales).astype(mx.int8)
        
        x_int8 = mx.reshape(x_int8, (-1, shape[-1]))
        x_int8 = x_int8[:n_samples, :]
        x_int8 = mx.reshape(x_int8, shape)
        scales = mx.reshape(scales, (n_blocks,))
        zero_points = mx.zeros_like(scales)
        return x_int8, scales, zero_points

    B, H, L, D = x.shape
    n_blocks = (L + block_size - 1) // block_size

    # Pad sequence length to multiple of block_size
    pad_len = n_blocks * block_size - L
    if pad_len > 0:
        x = mx.pad(x, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

    # Reshape to blocks: (B, H, n_blocks, block_size, D)
    x_blocks = mx.reshape(x, (B, H, n_blocks, block_size, D))

    # Apply sm_scale before quantization (matching NVIDIA kernel behavior)
    if is_q:
        # Q includes sm_scale * exp2 factor (1.44269504 = ln(2))
        x_blocks = x_blocks * (sm_scale * 1.44269504)
    else:
        # K uses sm_scale=1.0 (unscaled)
        pass

    # Symmetric quantization: use max(abs(x)) / 127 per block
    x_abs_max = mx.max(mx.abs(x_blocks), axis=(3, 4), keepdims=True)  # (B,H,n_blocks,1,1)
    scales = x_abs_max / 127.0
    scales = mx.where(scales == 0, mx.ones_like(scales) * 1e-6, scales)

    # Quantize with rounding (symmetric, so zero_point = 0)
    x_int8 = mx.round(x_blocks / scales).astype(mx.int8)

    # Reshape back to (B,H,L,D) and trim padding
    x_int8 = mx.reshape(x_int8, (B, H, n_blocks * block_size, D))
    if pad_len > 0:
        x_int8 = x_int8[:, :, :L, :]

    # Squeeze scales to (B, H, n_blocks); zero_points are zeros for symmetric quantization
    scales = mx.reshape(scales, (B, H, n_blocks))
    zero_points = mx.zeros((B, H, n_blocks), dtype=mx.float32)

    return x_int8, scales, zero_points


def _quantize_per_thread(x: mx.array, sm_scale: float = 1.0, is_q: bool = True) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Per-thread INT8 quantization using symmetric quantization.
    
    Quantizes along the head_dim dimension with finer granularity.
    """
    shape = x.shape
    
    # Apply sm_scale before quantization (NVIDIA aligned)
    if is_q:
        x = x * (sm_scale * 1.44269504)
    
    # Per-element symmetric scaling along head_dim
    x_abs_max = mx.max(mx.abs(x), axis=-1, keepdims=True)
    scale = x_abs_max / 127.0
    scale = mx.where(scale == 0, mx.ones_like(scale) * 1e-6, scale)
    
    # Quantize with rounding
    x_int8 = mx.round(x / scale).astype(mx.int8)
    
    # Zero points are zeros for symmetric quantization
    zero_point = mx.zeros_like(scale)
    
    return x_int8, scale, zero_point


def dequantize_int8(
    x_int8: mx.array,
    scale: mx.array,
    zero_point: mx.array
) -> mx.array:
    """
    Dequantize INT8 tensor back to float using symmetric quantization.
    
    For symmetric quantization, zero_point should be zeros.
    
    Parameters
    ----------
    x_int8 : mx.array
        Quantized int8 tensor
    scale : mx.array
        Quantization scale (max absolute / 127)
    zero_point : mx.array
        Quantization zero point (zeros for symmetric quantization)
        
    Returns
    -------
    x : mx.array
        Dequantized float tensor
    """
    # Expect scale, zero_point shapes: (B, H, n_blocks)
    if x_int8.ndim == 4 and scale.ndim == 3:
        B, H, L, D = x_int8.shape
        n_blocks = scale.shape[-1]
        block_size = (L + n_blocks - 1) // n_blocks

        # Reshape to (B,H,n_blocks,block_size,D)
        pad_len = n_blocks * block_size - L
        x = x_int8
        if pad_len > 0:
            x = mx.pad(x, ((0,0),(0,0),(0,pad_len),(0,0)))
        x_blocks = mx.reshape(x, (B, H, n_blocks, block_size, D))

        # Expand scales to (B,H,n_blocks,1,1)
        s = mx.reshape(scale, (B, H, n_blocks, 1, 1))

        # Symmetric dequantization: x = x_int8 * scale (zero_point is ignored)
        x_f = x_blocks.astype(mx.float32)
        x_deq = x_f * s
        x_deq = mx.reshape(x_deq, (B, H, n_blocks * block_size, D))
        if pad_len > 0:
            x_deq = x_deq[:, :, :L, :]
        return x_deq
    # Fallback for other shapes
    x = x_int8.astype(mx.float32)
    x = x * scale  # Symmetric: no zero_point subtraction
    return x


def quantize_pv(
    x: mx.array,
    dtype: str = "float8"
) -> Tuple[mx.array, mx.array]:
    """
    Quantize P×V result to FP8 or lower precision.
    
    Parameters
    ----------
    x : mx.array
        P×V tensor (attention_weights @ values)
    dtype : str
        Target dtype: "float8" or "float16"
        
    Returns
    -------
    x_quantized : mx.array
        Quantized tensor
    scale : mx.array
        Quantization scale
    """
    if dtype == "float8":
        return _quantize_float8(x)
    elif dtype == "float16":
        return x.astype(mx.float16), mx.array(1.0)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def _quantize_float8(x: mx.array) -> Tuple[mx.array, mx.array]:
    """
    FP8 quantization using amax scaling.
    
    Note: MLX doesn't natively support FP8, so we use float16 as proxy
    and store scale separately for potential custom kernel usage.
    """
    # For now, use float16 as MLX doesn't have native FP8
    # In production, this could be replaced with custom Metal kernels
    abs_max = mx.max(mx.abs(x))
    scale = abs_max / 240.0  # FP8 max value
    
    x_quantized = (x / scale).astype(mx.float16)
    
    return x_quantized, scale


def dequantize_fp8(
    x_fp8: mx.array,
    scale: mx.array
) -> mx.array:
    """
    Dequantize FP8 tensor back to float.
    
    Parameters
    ----------
    x_fp8 : mx.array
        Quantized float8 tensor (or float16 proxy)
    scale : mx.array
        Quantization scale
        
    Returns
    -------
    x : mx.array
        Dequantized float tensor
    """
    return x_fp8.astype(mx.float32) * scale
