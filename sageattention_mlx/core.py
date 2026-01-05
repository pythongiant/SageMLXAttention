"""
Core SageAttention MLX implementation for Apple Silicon.

This module provides optimized quantized attention mechanisms for Apple Silicon GPUs
using MLX as the backend.
"""

import mlx.core as mx
from typing import Optional, Tuple, Union, Dict
import math


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
    qk_quant_gran: str = "per_block",
    pv_dtype: str = "float16",
    smooth_k: bool = True,
    attn_mask: Optional[mx.array] = None,
    kv_cache: Optional[KVCache] = None,
    num_kv_heads: Optional[int] = None,
    tensor_format: str = "bhld",
    **kwargs
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    SageAttention: Quantized attention mechanism optimized for Apple Silicon.
    
    Implements INT8 quantization for Q×K^T computation and optional FP8 for P×V,
    maintaining accuracy while reducing computation and memory footprint.

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
        
    qk_quant_gran : str, optional
        Quantization granularity for Q×K: "per_block" or "per_thread".
        Default: "per_block".
        
    pv_dtype : str, optional
        Data type for P×V computation: "float16" or "float32".
        Default: "float16".
        
    smooth_k : bool, optional
        Whether to apply outlier smoothing to K. Default: True.
    
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
    if q.dtype not in [mx.float16, mx.float32]:
        raise ValueError(f"Query dtype must be float16 or float32, got {q.dtype}")
    
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
    
    # Apply outlier smoothing to K if requested
    if smooth_k:
        k = _smooth_tensor(k)
    
    # Perform attention computation with quantization
    output, lse = _quantized_attention(
        q, k, v,
        is_causal=is_causal,
        sm_scale=sm_scale,
        qk_quant_gran=qk_quant_gran,
        pv_dtype=pv_dtype,
        attn_mask=attn_mask
    )
    
    if return_lse:
        return output, lse
    return output


def _smooth_tensor(x: mx.array, eps: float = 1e-6) -> mx.array:
    """
    Apply variance-aware outlier smoothing along sequence dimension.
    
    Uses variance normalization to reduce the impact of outliers while
    preserving important signal information.
    
    Parameters
    ----------
    x : mx.array
        Input tensor
    eps : float
        Small epsilon to avoid division by zero
        
    Returns
    -------
    mx.array
        Smoothed tensor
    """
    # Compute mean and variance along sequence dimension (typically dim -2)
    mean = mx.mean(x, axis=-2, keepdims=True)
    var = mx.var(x, axis=-2, keepdims=True)
    
    # Normalize by standard deviation for variance-aware smoothing
    std = mx.sqrt(var + eps)
    smoothed = (x - mean) / std
    
    return smoothed


def _quantized_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    is_causal: bool,
    sm_scale: float,
    qk_quant_gran: str,
    pv_dtype: str,
    attn_mask: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """
    Core quantized attention computation using flash attention with streaming softmax.
    
    Parameters
    ----------
    q, k, v : mx.array
        Query, key, value tensors
    is_causal : bool
        Whether to apply causal mask
    sm_scale : float
        Softmax scale
    qk_quant_gran : str
        Quantization granularity
    pv_dtype : str
        P×V data type
    attn_mask : Optional[mx.array]
        Custom attention mask
        
    Returns
    -------
    output : mx.array
        Attention output
    lse : mx.array
        Log-sum-exp values
    """
    from .mlx_kernels import mlx_flash_attention_block

    # Preprocess custom attention mask into broadcastable (B, H, Lq, Lk) if provided
    if attn_mask is not None:
        if attn_mask.ndim == 3:
            if attn_mask.shape[0] == q.shape[0]:
                attn_mask = attn_mask[:, None, :, :]
            else:
                attn_mask = attn_mask[None, :, :, :]

        # Ensure mask has shape (B, H, Lq, Lk)
        if attn_mask.shape[0] != q.shape[0] or attn_mask.shape[1] != q.shape[1]:
            attn_mask = mx.broadcast_to(attn_mask, (q.shape[0], q.shape[1], q.shape[-2], k.shape[-2]))

    # Use flash attention with streaming softmax for memory efficiency
    output, lse = mlx_flash_attention_block(
        q, k, v,
        block_size=128,
        sm_scale=sm_scale,
        causal=is_causal,
        attn_mask=attn_mask,
    )
    
    return output, lse
