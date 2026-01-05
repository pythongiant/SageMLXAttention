"""
SageAttention MLX - Apple Silicon optimized attention using SageAttention algorithm.

This implementation provides the SageAttention algorithm optimized for Apple Silicon,
leveraging MLX's fast.scaled_dot_product_attention for maximum performance.

Key features:
- K smoothing via mean subtraction (NVIDIA SageAttention approach)
- LSE correction for ring attention compatibility
- Fused Metal kernel via mx.fast.scaled_dot_product_attention
- API compatibility with NVIDIA SageAttention
"""

from .core import sageattn_mlx, sageattn_qk_int8_pv_fp16_mlx, KVCache
from .quant import quantize_qk, quantize_pv, dequantize_int8, dequantize_fp8

__version__ = "0.2.0"

__all__ = [
    "sageattn_mlx",
    "sageattn_qk_int8_pv_fp16_mlx",
    "KVCache",
    "quantize_qk",
    "quantize_pv",
    "dequantize_int8",
    "dequantize_fp8",
]
