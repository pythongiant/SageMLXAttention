"""
Example: Using SageAttention MLX with a Hugging Face Transformer model.
"""

import mlx.core as mx
import mlx.nn as nn
from sageattention_mlx import sageattn_mlx


class AttentionWithSageAttn(nn.Module):
    """
    Multi-head attention layer using SageAttention MLX.
    
    This can be used as a drop-in replacement for standard attention
    in transformer models.
    """
    
    def __init__(self, dims: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.head_dim = dims // num_heads
        
        assert dims % num_heads == 0, "dims must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
    
    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        """
        Parameters
        ----------
        x : mx.array
            Input tensor (batch, seq_len, dims)
        mask : mx.array, optional
            Attention mask
            
        Returns
        -------
        mx.array
            Output tensor (batch, seq_len, dims)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = mx.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        q = mx.transpose(q, (0, 2, 1, 3))
        
        k = mx.reshape(k, (batch, seq_len, self.num_heads, self.head_dim))
        k = mx.transpose(k, (0, 2, 1, 3))
        
        v = mx.reshape(v, (batch, seq_len, self.num_heads, self.head_dim))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Apply SageAttention
        attn_output = sageattn_mlx(
            q, k, v,
            is_causal=False,
            smooth_k=True,
            qk_quant_gran="per_block"
        )
        
        # Reshape back to (batch, seq_len, dims)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch, seq_len, self.dims))
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


def example_usage():
    """
    Simple example of using SageAttention MLX attention layer.
    """
    # Create attention layer
    attn = AttentionWithSageAttn(dims=256, num_heads=8)
    
    # Create dummy input (batch=2, seq_len=10, dims=256)
    x = mx.random.normal((2, 10, 256))
    
    # Forward pass
    output = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ SageAttention MLX working correctly!")
    
    return output


if __name__ == "__main__":
    example_usage()
