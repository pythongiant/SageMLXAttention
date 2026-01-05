"""
Comprehensive benchmark comparing SageAttention MLX with baselines.

This script benchmarks:
1. Standard manual attention (matmul + softmax + matmul)
2. MLX native SDPA (mx.fast.scaled_dot_product_attention)
3. SageAttention MLX with smooth_k enabled
4. SageAttention MLX without smooth_k

Run with: PYTHONPATH=. python bench/benchmark_results.py
"""

import mlx.core as mx
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx import sageattn_mlx


def benchmark_single(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_iterations: int = 10,
    dtype=mx.float16,
):
    """Run benchmark for a single configuration."""
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    mx.eval(q, k, v)

    sm_scale = 1.0 / (head_dim ** 0.5)
    results = {}

    # 1. Standard manual attention
    # Warmup
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * sm_scale
    attn_weights = mx.softmax(scores, axis=-1)
    output = mx.matmul(attn_weights, v)
    mx.eval(output)

    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * sm_scale
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, v)
        mx.eval(output)
    results['manual'] = (time.perf_counter() - start) / num_iterations * 1000

    # 2. MLX native SDPA
    # Warmup
    output = mx.fast.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    mx.eval(output)

    start = time.perf_counter()
    for _ in range(num_iterations):
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        mx.eval(output)
    results['sdpa'] = (time.perf_counter() - start) / num_iterations * 1000

    # 3. SageAttention MLX with smooth_k
    # Warmup
    output = sageattn_mlx(q, k, v, smooth_k=True)
    mx.eval(output)

    start = time.perf_counter()
    for _ in range(num_iterations):
        output = sageattn_mlx(q, k, v, smooth_k=True)
        mx.eval(output)
    results['sageattn_smooth'] = (time.perf_counter() - start) / num_iterations * 1000

    # 4. SageAttention MLX without smooth_k
    # Warmup
    output = sageattn_mlx(q, k, v, smooth_k=False)
    mx.eval(output)

    start = time.perf_counter()
    for _ in range(num_iterations):
        output = sageattn_mlx(q, k, v, smooth_k=False)
        mx.eval(output)
    results['sageattn_no_smooth'] = (time.perf_counter() - start) / num_iterations * 1000

    return results


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 90)
    print("SageAttention MLX - Comprehensive Benchmark Results")
    print("=" * 90)
    print()
    print(f"MLX Version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Device: {mx.default_device()}")
    print()

    # Configuration
    batch_size = 2
    num_heads = 8
    head_dim = 64
    num_iterations = 10
    seq_lens = [256, 512, 1024, 2048, 4096, 8192]

    print(f"Configuration: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, dtype=float16")
    print(f"Iterations per test: {num_iterations}")
    print()

    # Header
    print("-" * 90)
    print(f"{'seq_len':<10} {'Manual':<12} {'SDPA':<12} {'SageAttn':<12} {'SA(no sm)':<12} {'vs Manual':<10} {'vs SDPA':<10}")
    print("-" * 90)

    all_results = []

    for seq_len in seq_lens:
        try:
            results = benchmark_single(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                num_iterations=num_iterations,
            )

            vs_manual = results['manual'] / results['sageattn_smooth']
            vs_sdpa = results['sdpa'] / results['sageattn_smooth']

            print(
                f"{seq_len:<10} "
                f"{results['manual']:>8.2f} ms  "
                f"{results['sdpa']:>8.2f} ms  "
                f"{results['sageattn_smooth']:>8.2f} ms  "
                f"{results['sageattn_no_smooth']:>8.2f} ms  "
                f"{vs_manual:>8.2f}x  "
                f"{vs_sdpa:>8.2f}x"
            )

            all_results.append({
                'seq_len': seq_len,
                **results,
                'vs_manual': vs_manual,
                'vs_sdpa': vs_sdpa,
            })

        except Exception as e:
            print(f"{seq_len:<10} Error: {e}")

    print("-" * 90)
    print()

    # Summary
    print("Legend:")
    print("  Manual     = Standard attention (matmul + softmax + matmul)")
    print("  SDPA       = mx.fast.scaled_dot_product_attention (MLX fused kernel)")
    print("  SageAttn   = sageattn_mlx with smooth_k=True")
    print("  SA(no sm)  = sageattn_mlx with smooth_k=False")
    print("  vs Manual  = Speedup of SageAttn over Manual (higher is better)")
    print("  vs SDPA    = Ratio of SDPA/SageAttn (>1 means SageAttn is faster)")
    print()

    # Analysis
    print("=" * 90)
    print("Analysis")
    print("=" * 90)
    print()

    if all_results:
        avg_vs_manual = sum(r['vs_manual'] for r in all_results) / len(all_results)
        avg_vs_sdpa = sum(r['vs_sdpa'] for r in all_results) / len(all_results)

        print(f"Average speedup vs Manual attention: {avg_vs_manual:.2f}x")
        print(f"Average ratio vs SDPA: {avg_vs_sdpa:.2f}x")
        print()

        if avg_vs_sdpa >= 0.95:
            print("SageAttention MLX achieves near-parity with MLX native SDPA!")
        if avg_vs_manual > 2.0:
            print(f"SageAttention MLX is {avg_vs_manual:.1f}x faster than manual attention!")

    return all_results


def run_accuracy_tests():
    """Run accuracy comparison tests."""
    print()
    print("=" * 90)
    print("Accuracy Tests")
    print("=" * 90)
    print()

    batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64

    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float32)
    mx.eval(q, k, v)

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Standard attention
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * sm_scale
    attn_weights = mx.softmax(scores, axis=-1)
    output_std = mx.matmul(attn_weights, v)
    mx.eval(output_std)

    # SageAttention without smooth_k
    output_sage_no_smooth = sageattn_mlx(q, k, v, smooth_k=False)
    mx.eval(output_sage_no_smooth)

    # SageAttention with smooth_k
    output_sage_smooth = sageattn_mlx(q, k, v, smooth_k=True)
    mx.eval(output_sage_smooth)

    # Compare
    diff_no_smooth = mx.abs(output_std - output_sage_no_smooth)
    diff_smooth = mx.abs(output_std - output_sage_smooth)

    print(f"Test configuration: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}")
    print()
    print("SageAttn (no smooth) vs Standard Attention:")
    print(f"  Max diff:  {mx.max(diff_no_smooth).item():.2e}")
    print(f"  Mean diff: {mx.mean(diff_no_smooth).item():.2e}")
    print()
    print("SageAttn (smooth_k) vs Standard Attention:")
    print(f"  Max diff:  {mx.max(diff_smooth).item():.2e}")
    print(f"  Mean diff: {mx.mean(diff_smooth).item():.2e}")
    print()

    # Check outputs are valid
    if mx.max(diff_no_smooth).item() < 1e-5:
        print("✓ SageAttn (no smooth) matches standard attention within tolerance")
    if mx.max(diff_smooth).item() < 1e-5:
        print("✓ SageAttn (smooth_k) matches standard attention within tolerance")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SageAttention MLX")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy tests")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarks")

    args = parser.parse_args()

    if not args.no_benchmark:
        run_benchmarks()

    if args.accuracy:
        run_accuracy_tests()

    print()
    print("=" * 90)
    print("Benchmark Complete")
    print("=" * 90)
