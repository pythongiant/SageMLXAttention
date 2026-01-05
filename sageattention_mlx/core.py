"""
Core SageAttention MLX implementation for Apple Silicon.

This module provides optimized attention mechanisms for Apple Silicon GPUs
using MLX as the backend, implementing the SageAttention algorithm.

Key optimizations from NVIDIA SageAttention:
1. K smoothing via mean subtraction (reduces outliers, improves quantization)
2. LSE correction when smooth_k is enabled
3. Optimized Metal kernel implementation via mlx_kernels
4. Optional quantized path for memory-bound scenarios
"""

import mlx.core as mx
from typing import Optional, Tuple, Union, Dict
import math
from .mlx_kernels import mlx_sage_attention, SageAttentionConfig


class KVCache:
    """Key-Value cache for efficient generation."""

    def __init__(self):
        self.k_cache: Optional[mx.array] = None
        self.v_cache: Optional[mx.array] = None

    def update(self, k: mx.array, v: mx.array):
        """Update cache with new K, V tensors."""
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = mx.concatenate([self.k_cache, k], axis=-2)
            self.v_cache = mx.concatenate([self.v_cache, v], axis=-2)

    def get(self) -> Tuple[mx.array, mx.array]:
        """Get cached K, V tensors."""
        return self.k_cache, self.v_cache

    def clear(self):
        """Clear the cache."""
        self.k_cache = None
        self.v_cache = None


def _convert_tensor_layout(
    x: mx.array,
    input_format: str,
    output_format: str
) -> mx.array:
    """
    Convert tensor layout between different formats.

    Parameters
    ----------
    x : mx.array
        Input tensor
    input_format : str
        Input layout: 'bhld' (batch, heads, length, dim) or 'bthn' (batch, length, heads, dim)
    output_format : str
        Output layout: 'bhld' or 'bthn'

    Returns
    -------
    mx.array
        Tensor in output format
    """
    if input_format == output_format:
        return x

    if input_format == "bthn" and output_format == "bhld":
        # (B, T, H, D) -> (B, H, T, D)
        return x.transpose(0, 2, 1, 3)
    elif input_format == "bhld" and output_format == "bthn":
        # (B, H, T, D) -> (B, T, H, D)
        return x.transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"Unsupported layout conversion: {input_format} -> {output_format}")


def _expand_kv_for_gqa(
    k: mx.array,
    v: mx.array,
    num_query_heads: int,
    num_kv_heads: int
) -> Tuple[mx.array, mx.array]:
    """
    Expand K, V tensors for group query attention.
    
    Replicates K, V heads to match number of query heads.
    
    Parameters
    ----------
    k : mx.array
        Key tensor (B, H_kv, L, D)
    v : mx.array
        Value tensor (B, H_kv, L, D)
    num_query_heads : int
        Number of query heads
    num_kv_heads : int
        Number of key/value heads
        
    Returns
    -------
    k_expanded : mx.array
        Expanded key tensor (B, H_q, L, D)
    v_expanded : mx.array
        Expanded value tensor (B, H_q, L, D)
    """
    if num_kv_heads == num_query_heads:
        return k, v
    
    # Repeat each K, V head num_query_heads // num_kv_heads times
    repeat_factor = num_query_heads // num_kv_heads
    
    # k, v shape: (B, H_kv, L, D)
    # Expand to (B, H_kv, L, D) -> (B, H_kv * repeat, L, D)
    k_expanded = mx.repeat(k, repeat_factor, axis=1)
    v_expanded = mx.repeat(v, repeat_factor, axis=1)
    
    return k_expanded, v_expanded


def sageattn_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    smooth_k: bool = True,
    attn_mask: Optional[mx.array] = None,
    kv_cache: Optional[KVCache] = None,
    num_kv_heads: Optional[int] = None,
    tensor_format: str = "bhld",
    use_fast_sdpa: bool = True,
    **kwargs
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    SageAttention: Optimized attention mechanism for Apple Silicon.

    Implements the SageAttention algorithm optimized for MLX, using:
    1. K smoothing via mean subtraction (NVIDIA-aligned approach)
    2. LSE correction for numerical accuracy when smooth_k is enabled
    3. mx.fast.scaled_dot_product_attention for fused Metal kernel

    Parameters
    ----------
    q : mx.array
        Query tensor of shape (batch, num_heads, seq_len_q, head_dim) in 'bhld' format
        or (batch, seq_len_q, num_heads, head_dim) in 'bthn' format.

    k : mx.array
        Key tensor of shape (batch, num_heads, seq_len_k, head_dim) or
        (batch, seq_len_k, num_heads, head_dim).

    v : mx.array
        Value tensor of shape (batch, num_heads, seq_len_k, head_dim) or
        (batch, seq_len_k, num_heads, head_dim).

    is_causal : bool, optional
        Whether to apply causal masking. Default: False.

    sm_scale : Optional[float], optional
        Scaling factor for softmax. If None, defaults to 1.0 / sqrt(head_dim).

    return_lse : bool, optional
        Whether to return log-sum-exp for ring attention compatibility.
        Default: False.

    smooth_k : bool, optional
        Whether to apply mean subtraction smoothing to K (NVIDIA SageAttention approach).
        This reduces outliers and improves numerical stability. Default: True.

    attn_mask : Optional[mx.array], optional
        Custom attention mask of shape (batch, seq_len_q, seq_len_k) or
        (num_heads, seq_len_q, seq_len_k). True values are masked (set to -inf).
        Default: None.

    kv_cache : Optional[KVCache], optional
        K-V cache for efficient generation. If provided, uses cached K, V
        and updates with new key/value tensors. Default: None.

    num_kv_heads : Optional[int], optional
        Number of K/V heads for group query attention. If provided and less than
        num_query_heads, implements GQA by repeating K/V heads. Default: None.

    tensor_format : str, optional
        Input tensor layout: 'bhld' (batch, heads, length, dim) or
        'bthn' (batch, length, heads, dim). Default: 'bhld'.

    use_fast_sdpa : bool, optional
        Whether to use mx.fast.scaled_dot_product_attention (recommended).
        Default: True.

    Returns
    -------
    mx.array
        Output attention tensor of same shape as query.

    Tuple[mx.array, mx.array]
        If return_lse=True, returns (output, lse).
    """

    # Convert tensor layout if needed
    if tensor_format == "bthn":
        q = _convert_tensor_layout(q, "bthn", "bhld")
        k = _convert_tensor_layout(k, "bthn", "bhld")
        v = _convert_tensor_layout(v, "bthn", "bhld")

    # Input validation
    if q.dtype not in [mx.float16, mx.float32, mx.bfloat16]:
        raise ValueError(f"Query dtype must be float16, bfloat16, or float32, got {q.dtype}")

    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("Q, K, V must have the same dtype")

    # Compute scaling factor
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Handle KV cache
    if kv_cache is not None:
        kv_cache.update(k, v)
        k, v = kv_cache.get()

    # Handle group query attention
    if num_kv_heads is not None:
        num_query_heads = q.shape[1]
        k, v = _expand_kv_for_gqa(k, v, num_query_heads, num_kv_heads)

    # Apply K smoothing via mean subtraction (NVIDIA SageAttention approach)
    # This reduces outliers in K which improves quantization accuracy
    km = None
    lse_correction = None
    if smooth_k:
        # Compute mean along sequence dimension: (B, H, 1, D)
        km = mx.mean(k, axis=-2, keepdims=True)
        # Subtract mean from K
        k = k - km

        # Compute LSE correction term: Q @ km^T * sm_scale
        # This corrects for the mean subtraction in the final LSE
        if return_lse:
            # km: (B, H, 1, D), need (B, H, D, 1) for matmul
            km_t = km.transpose(0, 1, 3, 2)  # (B, H, D, 1)
            # Q @ km^T: (B, H, Lq, D) @ (B, H, D, 1) -> (B, H, Lq, 1)
            lse_correction = mx.matmul(q, km_t).squeeze(-1)  # (B, H, Lq)
            lse_correction = lse_correction.astype(mx.float32)

    # Perform attention computation using optimized Metal kernels
    if use_fast_sdpa and not return_lse and attn_mask is None:
        # Use optimized streaming kernel from mlx_kernels
        cfg = SageAttentionConfig(sm_scale=sm_scale if sm_scale != 1.0 / math.sqrt(head_dim) else None)
        output, lse = mlx_sage_attention(q, k, v, cfg=cfg)
        # Convert output back to input dtype
        output = output.astype(q.dtype)
        if not return_lse:
            lse = None
    else:
        # Use manual attention for LSE or custom masks
        output, lse = _attention_with_lse(
            q, k, v,
            is_causal=is_causal,
            sm_scale=sm_scale,
            attn_mask=attn_mask
        )

    # Apply LSE correction for smooth_k
    if return_lse and lse is not None and lse_correction is not None:
        lse = lse + lse_correction * sm_scale

    if return_lse:
        return output, lse
    return output


def _fast_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    sm_scale: float,
    is_causal: bool = False
) -> mx.array:
    """
    Fast attention using optimized Metal kernels.

    This is the recommended path for most use cases as it uses
    streaming softmax kernels optimized for Apple Silicon.
    """
    cfg = SageAttentionConfig(sm_scale=sm_scale)
    output, _ = mlx_sage_attention(q, k, v, cfg=cfg)
    return output


def _attention_with_lse(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    is_causal: bool,
    sm_scale: float,
    attn_mask: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """
    Attention computation that returns LSE for ring attention support.

    Uses streaming softmax for numerical stability.
    """
    # Compute attention scores: (B, H, Lq, Lk)
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * sm_scale

    # Apply causal mask if needed
    if is_causal:
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        # Create causal mask (True = mask out)
        row_idx = mx.arange(seq_len_q)[:, None]
        col_idx = mx.arange(seq_len_k)[None, :]
        # For causal: mask positions where col > row (future tokens)
        # Need to handle case where seq_len_q != seq_len_k (e.g., during generation)
        offset = seq_len_k - seq_len_q
        causal_mask = col_idx > (row_idx + offset)
        scores = mx.where(causal_mask, mx.array(-float('inf')), scores)

    # Apply custom attention mask if provided
    if attn_mask is not None:
        # Expand mask to (B, H, Lq, Lk) if needed
        if attn_mask.ndim == 3:
            if attn_mask.shape[0] == q.shape[0]:
                attn_mask = attn_mask[:, None, :, :]  # (B, 1, Lq, Lk)
            else:
                attn_mask = attn_mask[None, :, :, :]  # (1, H, Lq, Lk)
        scores = mx.where(attn_mask, mx.array(-float('inf')), scores)

    # Compute LSE for each query position: (B, H, Lq)
    lse = mx.logsumexp(scores, axis=-1)

    # Compute softmax attention weights
    attn_weights = mx.softmax(scores, axis=-1)

    # Compute output: (B, H, Lq, D)
    output = mx.matmul(attn_weights, v)

    return output, lse


# Legacy function kept for backward compatibility with existing code
def sageattn_qk_int8_pv_fp16_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    MLX equivalent of sageattn_qk_int8_pv_fp16_triton from NVIDIA SageAttention.

    This function provides API compatibility with the NVIDIA implementation
    while using the optimized MLX backend.

    Parameters match the NVIDIA sageattn_qk_int8_pv_fp16_triton function.
    """
    # Convert tensor_layout to tensor_format
    tensor_format = "bhld" if tensor_layout == "HND" else "bthn"

    return sageattn_mlx(
        q, k, v,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        smooth_k=smooth_k,
        tensor_format=tensor_format,
        use_fast_sdpa=True,
        **kwargs
    )
