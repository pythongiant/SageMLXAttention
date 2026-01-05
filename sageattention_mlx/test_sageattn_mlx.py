"""
Unit tests for SageAttention MLX.
"""

import mlx.core as mx
import pytest
import sys
import os

# Handle both relative imports (when run as pytest) and direct script execution
try:
    from .core import sageattn_mlx, KVCache
    from .quant import quantize_qk, quantize_pv
    from .quant import dequantize_int8, dequantize_fp8
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(__file__))
    from core import sageattn_mlx, KVCache
    from quant import quantize_qk, quantize_pv
    from quant import dequantize_int8, dequantize_fp8


class TestQuantization:
    """Test quantization utilities."""
    
    def test_quantize_qk_per_block(self):
        """Test per-block INT8 quantization for Q/K."""
        x = mx.random.normal((2, 8, 128, 64))
        q_int8, scale, zp = quantize_qk(x, gran="per_block")
        
        assert q_int8.dtype == mx.int8
        assert q_int8.shape == x.shape
    
    def test_quantize_qk_per_thread(self):
        """Test per-thread INT8 quantization for Q/K."""
        x = mx.random.normal((2, 8, 128, 64))
        q_int8, scale, zp = quantize_qk(x, gran="per_thread")
        
        assert q_int8.dtype == mx.int8
        assert q_int8.shape == x.shape
    
    def test_dequantize_int8(self):
        """Test INT8 dequantization."""
        x = mx.random.normal((2, 8, 64))
        q_int8, scale, zp = quantize_qk(x, gran="per_thread")
        
        x_recovered = dequantize_int8(q_int8, scale, zp)
        assert x_recovered.dtype == mx.float32
        assert x_recovered.shape == x.shape


class TestAttention:
    """Test attention computation."""
    
    def test_sageattn_basic(self):
        """Test basic SageAttention forward pass."""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v)
        
        assert output.shape == (batch, heads, seq_len, head_dim)
        assert output.dtype == mx.float16
    
    def test_sageattn_with_lse(self):
        """Test SageAttention with LSE output."""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        output, lse = sageattn_mlx(q, k, v, return_lse=True)
        
        assert output.shape == (batch, heads, seq_len, head_dim)
        assert lse.shape == (batch, heads, seq_len)
    
    def test_sageattn_causal(self):
        """Test SageAttention with causal masking."""
        batch, heads, seq_len, head_dim = 2, 8, 32, 64
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v, is_causal=True)
        
        assert output.shape == (batch, heads, seq_len, head_dim)
    
    def test_sageattn_different_dtypes(self):
        """Test SageAttention with different data types."""
        batch, heads, seq_len, head_dim = 2, 8, 32, 64
        
        for dtype in [mx.float16, mx.float32]:
            q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=dtype)
            k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=dtype)
            v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=dtype)
            
            output = sageattn_mlx(q, k, v)
            assert output.dtype == dtype


class TestQuantizationGranularity:
    """Test different quantization granularities."""
    
    def test_per_block_vs_per_thread(self):
        """Compare per-block and per-thread quantization."""
        x = mx.random.normal((2, 8, 128, 64))
        
        q_pb, _, _ = quantize_qk(x, gran="per_block")
        q_pt, _, _ = quantize_qk(x, gran="per_thread")
        
        # Both should be valid int8 tensors
        assert q_pb.dtype == mx.int8
        assert q_pt.dtype == mx.int8
        
        # Shapes should match
        assert q_pb.shape == q_pt.shape == x.shape


class TestKVCache:
    """Test KV cache functionality."""
    
    def test_kv_cache_basic(self):
        """Test basic KV cache operations."""
        cache = KVCache()
        
        k1 = mx.random.normal((2, 8, 10, 64))
        v1 = mx.random.normal((2, 8, 10, 64))
        
        cache.update(k1, v1)
        k_cached, v_cached = cache.get()
        
        assert mx.allclose(k_cached, k1)
        assert mx.allclose(v_cached, v1)
    
    def test_kv_cache_append(self):
        """Test KV cache appending."""
        cache = KVCache()
        
        k1 = mx.random.normal((2, 8, 10, 64))
        v1 = mx.random.normal((2, 8, 10, 64))
        k2 = mx.random.normal((2, 8, 5, 64))
        v2 = mx.random.normal((2, 8, 5, 64))
        
        cache.update(k1, v1)
        cache.update(k2, v2)
        
        k_cached, v_cached = cache.get()
        
        # Cached K should have length 15 (10 + 5)
        assert k_cached.shape[-2] == 15
        assert v_cached.shape[-2] == 15
    
    def test_sageattn_with_kv_cache(self):
        """Test SageAttention with KV cache."""
        batch, heads, head_dim = 2, 8, 64
        seq_len_new = 5
        
        q = mx.random.normal((batch, heads, seq_len_new, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len_new, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len_new, head_dim), dtype=mx.float16)
        
        cache = KVCache()
        output = sageattn_mlx(q, k, v, kv_cache=cache)
        
        assert output.shape == (batch, heads, seq_len_new, head_dim)


class TestTensorLayout:
    """Test tensor layout conversion."""
    
    def test_sageattn_bthn_format(self):
        """Test SageAttention with BTHN tensor format."""
        batch, seq_len, heads, head_dim = 2, 64, 8, 64
        
        q = mx.random.normal((batch, seq_len, heads, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, seq_len, heads, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, seq_len, heads, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v, tensor_format="bthn")
        
        # Output should be in BHLD format internally, but shape should match input conceptually
        assert output.shape == (batch, heads, seq_len, head_dim)
    
    def test_sageattn_bhld_format(self):
        """Test SageAttention with BHLD tensor format (default)."""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64
        
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v, tensor_format="bhld")
        
        assert output.shape == (batch, heads, seq_len, head_dim)


class TestAttentionMask:
    """Test attention mask functionality."""
    
    def test_sageattn_with_custom_mask(self):
        """Test SageAttention with custom attention mask."""
        batch, heads, seq_len, head_dim = 2, 8, 32, 64
        
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        # Create a mask where last 5 tokens are masked
        attn_mask = mx.zeros((batch, seq_len, seq_len), dtype=mx.bool_)
        attn_mask[:, :, -5:] = True  # Mask last 5 positions
        
        output = sageattn_mlx(q, k, v, attn_mask=attn_mask)
        
        assert output.shape == (batch, heads, seq_len, head_dim)
    
    def test_sageattn_mask_shape_compatibility(self):
        """Test attention mask shape compatibility."""
        batch, heads, seq_len, head_dim = 2, 8, 32, 64
        
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        # Test with per-head mask
        attn_mask = mx.zeros((heads, seq_len, seq_len), dtype=mx.bool_)
        output = sageattn_mlx(q, k, v, attn_mask=attn_mask)
        
        assert output.shape == (batch, heads, seq_len, head_dim)


class TestGroupQueryAttention:
    """Test group query attention functionality."""
    
    def test_sageattn_gqa(self):
        """Test SageAttention with group query attention."""
        batch, heads_q, heads_kv, seq_len, head_dim = 2, 8, 2, 64, 64
        
        q = mx.random.normal((batch, heads_q, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v, num_kv_heads=heads_kv)
        
        assert output.shape == (batch, heads_q, seq_len, head_dim)
    
    def test_sageattn_gqa_full_heads(self):
        """Test GQA with full number of heads (no reduction)."""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64
        
        q = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads, seq_len, head_dim), dtype=mx.float16)
        
        # With num_kv_heads == num_query_heads, should work normally
        output = sageattn_mlx(q, k, v, num_kv_heads=heads)
        
        assert output.shape == (batch, heads, seq_len, head_dim)


class TestCombinedFeatures:
    """Test combinations of multiple features."""
    
    def test_sageattn_gqa_with_causal(self):
        """Test GQA with causal masking."""
        batch, heads_q, heads_kv, seq_len, head_dim = 2, 8, 2, 32, 64
        
        q = mx.random.normal((batch, heads_q, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        
        output = sageattn_mlx(q, k, v, is_causal=True, num_kv_heads=heads_kv)
        
        assert output.shape == (batch, heads_q, seq_len, head_dim)
    
    def test_sageattn_gqa_with_mask_and_cache(self):
        """Test GQA with custom mask and KV cache."""
        batch, heads_q, heads_kv, seq_len, head_dim = 2, 8, 2, 32, 64
        
        q = mx.random.normal((batch, heads_q, seq_len, head_dim), dtype=mx.float16)
        k = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        v = mx.random.normal((batch, heads_kv, seq_len, head_dim), dtype=mx.float16)
        
        attn_mask = mx.zeros((batch, seq_len, seq_len), dtype=mx.bool_)
        cache = KVCache()
        
        output = sageattn_mlx(
            q, k, v,
            num_kv_heads=heads_kv,
            attn_mask=attn_mask,
            kv_cache=cache
        )
        
        assert output.shape == (batch, heads_q, seq_len, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
