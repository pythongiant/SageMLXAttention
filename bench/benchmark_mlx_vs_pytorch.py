"""
Comprehensive benchmark: SageAttention MLX vs PyTorch MPS vs Standard MLX attention.

Compare performance, memory usage, and speedup factors across different backends.
"""

import mlx.core as mx
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx import sageattn_mlx

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Skipping PyTorch MPS benchmarks.")


def benchmark_mlx_standard(q, k, v, num_iterations=10, head_dim=64):
    """Benchmark standard MLX attention."""
    mx.eval(q, k, v)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / (head_dim ** 0.5)
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, v)
        mx.eval(output)
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations


def benchmark_sageattn_mlx(q, k, v, num_iterations=10):
    """Benchmark SageAttention MLX."""
    mx.eval(q, k, v)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = sageattn_mlx(q, k, v)
        mx.eval(output)
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations


def benchmark_pytorch_mps(q_pt, k_pt, v_pt, num_iterations=10, head_dim=64):
    """Benchmark PyTorch MPS attention."""
    if not PYTORCH_AVAILABLE:
        return None
    
    # Warm up
    with torch.no_grad():
        for _ in range(2):
            scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v_pt)
    
    torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v_pt)
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iterations


def get_pytorch_memory_allocated():
    """Get PyTorch MPS allocated memory in MB."""
    if PYTORCH_AVAILABLE:
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0


def benchmark_suite(
    batch_size=2,
    num_heads=8,
    seq_len=512,
    head_dim=64,
    num_iterations=10,
):
    """Run complete benchmark suite."""
    
    print("=" * 80)
    print("SageAttention MLX vs PyTorch MPS Benchmark")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Num Heads:     {num_heads}")
    print(f"  Sequence Len:  {seq_len}")
    print(f"  Head Dim:      {head_dim}")
    print(f"  Total Shape:   ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
    print(f"  Iterations:    {num_iterations}")
    print(f"  Total Flops:   {2 * batch_size * num_heads * seq_len * seq_len * head_dim / 1e9:.2f}B")
    
    # Create MLX test data
    q_mlx = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    k_mlx = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    v_mlx = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
    
    # Convert to float32 for computation stability
    q_mlx = q_mlx.astype(mx.float32)
    k_mlx = k_mlx.astype(mx.float32)
    v_mlx = v_mlx.astype(mx.float32)
    
    print("\n" + "=" * 80)
    print("MLX Benchmarks")
    print("=" * 80)
    
    # Benchmark standard MLX
    print(f"\nStandard MLX Attention...")
    mlx_std_time = benchmark_mlx_standard(q_mlx, k_mlx, v_mlx, num_iterations, head_dim)
    print(f"  Time: {mlx_std_time*1000:.3f} ms")
    print(f"  TFLOPS: {2 * batch_size * num_heads * seq_len * seq_len * head_dim / (mlx_std_time * 1e12):.2f}")
    
    # Benchmark SageAttention MLX
    print(f"\nSageAttention MLX...")
    sageattn_time = benchmark_sageattn_mlx(q_mlx, k_mlx, v_mlx, num_iterations)
    print(f"  Time: {sageattn_time*1000:.3f} ms")
    print(f"  TFLOPS: {2 * batch_size * num_heads * seq_len * seq_len * head_dim / (sageattn_time * 1e12):.2f}")
    
    speedup_mlx = mlx_std_time / sageattn_time
    print(f"\n  Speedup (MLX std vs SageAttention): {speedup_mlx:.2f}x")
    print(f"  Performance Improvement: {(1 - sageattn_time/mlx_std_time) * 100:.1f}%")
    
    # PyTorch MPS benchmarks
    if PYTORCH_AVAILABLE:
        print("\n" + "=" * 80)
        print("PyTorch MPS Benchmarks")
        print("=" * 80)
        
        # Create PyTorch test data
        q_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device='mps')
        k_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device='mps')
        v_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device='mps')
        
        print(f"\nPyTorch MPS Standard Attention...")
        pytorch_time = benchmark_pytorch_mps(q_pt, k_pt, v_pt, num_iterations, head_dim)
        print(f"  Time: {pytorch_time*1000:.3f} ms")
        print(f"  TFLOPS: {2 * batch_size * num_heads * seq_len * seq_len * head_dim / (pytorch_time * 1e12):.2f}")
        
        # Comparison
        print("\n" + "=" * 80)
        print("Cross-Backend Comparison")
        print("=" * 80)
        
        print(f"\nExecution Time (ms):")
        print(f"  {'Benchmark':<30} {'Time (ms)':<15} {'vs SageAttention'}")
        print(f"  {'-'*55}")
        print(f"  {'Standard MLX':<30} {mlx_std_time*1000:>8.3f}        {mlx_std_time/sageattn_time:>6.2f}x")
        print(f"  {'SageAttention MLX':<30} {sageattn_time*1000:>8.3f}        {'1.00x':>6}")
        print(f"  {'PyTorch MPS':<30} {pytorch_time*1000:>8.3f}        {pytorch_time/sageattn_time:>6.2f}x")
        
        speedup_pytorch = pytorch_time / sageattn_time
        print(f"\nPerformance Analysis:")
        print(f"  SageAttention vs MLX Standard:  {speedup_mlx:.2f}x faster")
        print(f"  SageAttention vs PyTorch MPS:  {speedup_pytorch:.2f}x {'faster' if speedup_pytorch > 1 else 'slower'}")
        print(f"  MLX Standard vs PyTorch MPS:   {mlx_std_time/pytorch_time:.2f}x {'faster' if mlx_std_time < pytorch_time else 'slower'}")
        
        # Determine best implementation
        times = {
            "Standard MLX": mlx_std_time,
            "SageAttention MLX": sageattn_time,
            "PyTorch MPS": pytorch_time,
        }
        best = min(times, key=times.get)
        print(f"\n  🏆 Best Performance: {best}")
        
    else:
        print("\n" + "=" * 80)
        print("PyTorch not available - skipping MPS benchmarks")
        print("=" * 80)
        print("\nTo compare with PyTorch MPS:")
        print("  pip install torch torchvision torchaudio")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark SageAttention MLX vs PyTorch MPS")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    
    benchmark_suite(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_iterations=args.iterations,
    )
