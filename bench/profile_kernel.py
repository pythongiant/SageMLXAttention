"""
Detailed profiling of mlx_flash_attention_block to identify performance bottlenecks.
"""

import mlx.core as mx
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx.mlx_kernels import mlx_flash_attention_block


def profile_attention_block(
    batch_size=2,
    num_heads=8,
    seq_len=512,
    head_dim=64,
    block_size=128,
    num_iterations=5,
):
    """Profile mlx_flash_attention_block to identify bottlenecks."""
    
    print("=" * 90)
    print("mlx_flash_attention_block Profiling")
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
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    
    mx.eval(q, k, v)
    
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print("\n" + "=" * 90)
    print("Test 1: Standard Attention (use_blockwise_quant=False)")
    print("=" * 90)
    
    times_standard = []
    for i in range(num_iterations):
        start = time.perf_counter()
        output, lse = mlx_flash_attention_block(
            q, k, v,
            block_size=block_size,
            sm_scale=sm_scale,
            causal=False,
            use_blockwise_quant=False,
        )
        mx.eval(output, lse)
        elapsed = time.perf_counter() - start
        times_standard.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.3f} ms")
    
    avg_standard = np.mean(times_standard)
    std_standard = np.std(times_standard)
    print(f"  Average: {avg_standard*1000:.3f} ± {std_standard*1000:.3f} ms")
    
    print("\n" + "=" * 90)
    print("Test 2: Blockwise Quantized Attention (use_blockwise_quant=True)")
    print("=" * 90)
    
    times_quant = []
    for i in range(num_iterations):
        start = time.perf_counter()
        output, lse = mlx_flash_attention_block(
            q, k, v,
            block_size=block_size,
            sm_scale=sm_scale,
            causal=False,
            use_blockwise_quant=True,
        )
        mx.eval(output, lse)
        elapsed = time.perf_counter() - start
        times_quant.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.3f} ms")
    
    avg_quant = np.mean(times_quant)
    std_quant = np.std(times_quant)
    print(f"  Average: {avg_quant*1000:.3f} ± {std_quant*1000:.3f} ms")
    
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Standard Attention:        {avg_standard*1000:.3f} ms (baseline)")
    print(f"Quantized Attention:       {avg_quant*1000:.3f} ms")
    print(f"Overhead:                  {(avg_quant/avg_standard - 1)*100:.1f}%")
    print(f"Speedup:                   {avg_standard/avg_quant:.2f}x")
    
    if avg_quant < avg_standard:
        print(f"✅ Quantized is FASTER by {(avg_standard - avg_quant)*1000:.3f} ms")
    else:
        print(f"❌ Quantized is SLOWER by {(avg_quant - avg_standard)*1000:.3f} ms")


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
