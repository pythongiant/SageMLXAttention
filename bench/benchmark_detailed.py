"""
Detailed benchmark showing optimization components and their impact.

Breaks down the performance of SageAttention MLX to understand where improvements come from.
"""

import mlx.core as mx
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx import sageattn_mlx, quantize_qk, dequantize_int8


def benchmark_with_breakdown(
    batch_size=2,
    num_heads=8,
    seq_len=512,
    head_dim=64,
    num_iterations=10,
):
    """Benchmark with detailed breakdown of operations."""
    
    print("=" * 90)
    print("SageAttention MLX - Component Performance Breakdown")
    print("=" * 90)
    
    print(f"\nConfiguration:")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Num Heads:     {num_heads}")
    print(f"  Sequence Len:  {seq_len}")
    print(f"  Head Dim:      {head_dim}")
    print(f"  Total Shape:   ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
    print(f"  Iterations:    {num_iterations}")
    
    # Create test data
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    
    mx.eval(q, k, v)
    
    print("\n" + "=" * 90)
    print("Individual Component Timing")
    print("=" * 90)
    
    # Benchmark 1: Standard attention (baseline)
    print(f"\n1. Standard MLX Attention (Baseline)")
    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / (head_dim ** 0.5)
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, v)
        mx.eval(output)
    std_time = (time.perf_counter() - start) / num_iterations
    print(f"   Total Time: {std_time*1000:.3f} ms")
    
    # Benchmark 2: Quantization step
    print(f"\n2. Quantization to INT8 (Per-block)")
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_int8, q_scale, q_zp = quantize_qk(q, gran="per_block")
        k_int8, k_scale, k_zp = quantize_qk(k, gran="per_block")
        mx.eval(q_int8, k_int8)
    quant_time = (time.perf_counter() - start) / num_iterations
    print(f"   Total Time: {quant_time*1000:.3f} ms")
    print(f"   Breakdown per tensor:")
    print(f"     - Q quantization: ~{quant_time/2*1000:.3f} ms")
    print(f"     - K quantization: ~{quant_time/2*1000:.3f} ms")
    
    # Benchmark 3: Dequantization (simulated - use per_thread for simpler scale shape)
    print(f"\n3. Dequantization from INT8")
    from sageattention_mlx.quant import _quantize_per_thread
    q_int8_pt, q_scale_pt, q_zp_pt = _quantize_per_thread(q)
    k_int8_pt, k_scale_pt, k_zp_pt = _quantize_per_thread(k)
    mx.eval(q_int8_pt, k_int8_pt, q_scale_pt, q_zp_pt, k_scale_pt, k_zp_pt)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_dequant = dequantize_int8(q_int8_pt, q_scale_pt, q_zp_pt)
        k_dequant = dequantize_int8(k_int8_pt, k_scale_pt, k_zp_pt)
        mx.eval(q_dequant, k_dequant)
    dequant_time = (time.perf_counter() - start) / num_iterations
    print(f"   Total Time: {dequant_time*1000:.3f} ms")
    
    # Benchmark 4: Quantized matmul
    print(f"\n4. Attention Computation (Quantized)")
    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = mx.matmul(q_int8.astype(mx.float32), 
                          mx.transpose(k_int8.astype(mx.float32), (0, 1, 3, 2))) / (head_dim ** 0.5)
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, v)
        mx.eval(output)
    quant_attn_time = (time.perf_counter() - start) / num_iterations
    print(f"   Total Time: {quant_attn_time*1000:.3f} ms")
    
    # Benchmark 5: Full SageAttention
    print(f"\n5. Full SageAttention MLX")
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = sageattn_mlx(q, k, v)
        mx.eval(output)
    sageattn_time = (time.perf_counter() - start) / num_iterations
    print(f"   Total Time: {sageattn_time*1000:.3f} ms")
    
    # Summary
    print("\n" + "=" * 90)
    print("Performance Summary")
    print("=" * 90)
    
    print(f"\n{'Benchmark':<40} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'Standard MLX Attention':<40} {std_time*1000:>8.3f}        {'1.00x':>8}")
    print(f"{'Quantization (both Q,K)':<40} {quant_time*1000:>8.3f}        {std_time/quant_time:>8.2f}x")
    print(f"{'Dequantization (both Q,K)':<40} {dequant_time*1000:>8.3f}        {std_time/dequant_time:>8.2f}x")
    print(f"{'Attention with Quantized Q,K':<40} {quant_attn_time*1000:>8.3f}        {std_time/quant_attn_time:>8.2f}x")
    print(f"{'Full SageAttention MLX':<40} {sageattn_time*1000:>8.3f}        {std_time/sageattn_time:>8.2f}x")
    
    # Analysis
    print(f"\nOptimization Impact Analysis:")
    print(f"  Quantization Overhead: {(quant_time + dequant_time)*1000:.3f} ms")
    print(f"  Total Overhead: {((quant_time + dequant_time + sageattn_time) - std_time)*1000:.3f} ms")
    print(f"  Efficiency: {(std_time - sageattn_time) / (quant_time + dequant_time) * 100:.1f}%")
    
    print("\n" + "=" * 90)
    print("Benchmark Complete")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed SageAttention breakdown")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    
    benchmark_with_breakdown(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        num_iterations=args.iterations,
    )
