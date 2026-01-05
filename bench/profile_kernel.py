"""
Detailed profiling of optimized SageAttention Metal kernels.
"""

import mlx.core as mx
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx.mlx_kernels import mlx_sage_attention, SageAttentionConfig


def profile_attention_block(
    batch_size=2,
    num_heads=8,
    seq_len=512,
    head_dim=64,
    block_size=128,
    num_iterations=5,
):
    """Profile mlx_sage_attention kernel to identify performance characteristics."""
    
    print("=" * 90)
    print("SageAttention Metal Kernel Profiling")
    print("=" * 90)
    
    print(f"\nConfiguration:")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Num Heads:      {num_heads}")
    print(f"  Sequence Len:   {seq_len}")
    print(f"  Head Dim:       {head_dim}")
    print(f"  Block Size:     {block_size}")
    print(f"  Total Shape:    ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
    print(f"  Iterations:     {num_iterations}")
    
    # Create test data
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    
    mx.eval(q, k, v)
    
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print("\n" + "=" * 90)
    print("Test: Optimized Streaming Attention (mlx_sage_attention)")
    print("=" * 90)
    
    times = []
    cfg = SageAttentionConfig(block_q=block_size, block_k=block_size, sm_scale=sm_scale)
    
    for i in range(num_iterations):
        start = time.perf_counter()
        output, lse = mlx_sage_attention(q, k, v, cfg=cfg)
        mx.eval(output, lse)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.3f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Average: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Throughput:  {batch_size * num_heads * seq_len * seq_len * head_dim / (avg_time * 1e9):.2f} GFLOPS")
    print(f"Memory BW:   {batch_size * num_heads * seq_len * head_dim * 3 * 2 / (avg_time * 1e9):.2f} GB/s")


if __name__ == "__main__":
    # Profile with different sequence lengths
    print("\n\n")
    print("╔" + "=" * 88 + "╗")
    print("║" + " " * 88 + "║")
    print("║" + "PROFILE 1: seq_len=256 (small)".center(88) + "║")
    print("║" + " " * 88 + "║")
    print("╚" + "=" * 88 + "╝")
    profile_attention_block(seq_len=256, block_size=64, num_iterations=5)
    
    print("\n\n")
    print("╔" + "=" * 88 + "╗")
    print("║" + " " * 88 + "║")
    print("║" + "PROFILE 2: seq_len=512 (medium)".center(88) + "║")
    print("║" + " " * 88 + "║")
    print("╚" + "=" * 88 + "╝")
    profile_attention_block(seq_len=512, block_size=128, num_iterations=5)
    
    print("\n\n")
    print("╔" + "=" * 88 + "╗")
    print("║" + " " * 88 + "║")
    print("║" + "PROFILE 3: seq_len=1024 (large)".center(88) + "║")
    print("║" + " " * 88 + "║")
    print("╚" + "=" * 88 + "╝")
    profile_attention_block(seq_len=1024, block_size=128, num_iterations=3)
