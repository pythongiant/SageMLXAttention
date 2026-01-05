"""
Benchmark script for SageAttention MLX.

Compare performance with standard MLX attention.
"""

import mlx.core as mx
import time
from sageattention_mlx import sageattn_mlx


def benchmark_attention(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    num_iterations: int = 10,
):
    """
    Benchmark SageAttention MLX vs standard attention.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    num_heads : int
        Number of attention heads
    seq_len : int
        Sequence length
    head_dim : int
        Head dimension
    num_iterations : int
        Number of iterations for averaging
    """
    
    # Create test data
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    
    mx.eval(q, k, v)
    
    # Benchmark standard attention
    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / (head_dim ** 0.5)
        attn_weights = mx.softmax(scores, axis=-1)
        output_std = mx.matmul(attn_weights, v)
        mx.eval(output_std)
    std_time = (time.perf_counter() - start) / num_iterations
    
    # Benchmark SageAttention MLX
    start = time.perf_counter()
    for _ in range(num_iterations):
        output_sage = sageattn_mlx(q, k, v)
        mx.eval(output_sage)
    sage_time = (time.perf_counter() - start) / num_iterations
    
    # Print results
    print(f"\nBenchmark Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Seq Len: {seq_len}")
    print(f"  Head Dim: {head_dim}")
    print(f"  Total Shape: ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
    print(f"  Iterations: {num_iterations}")
    
    print(f"\nResults:")
    print(f"  Standard Attention: {std_time*1000:.2f} ms")
    print(f"  SageAttention MLX:  {sage_time*1000:.2f} ms")
    print(f"  Speedup: {std_time/sage_time:.2f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark SageAttention MLX")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    
    benchmark_attention(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_iterations=args.iterations,
    )
