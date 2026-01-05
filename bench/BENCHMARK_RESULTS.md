# SageAttention MLX Benchmark Results

## Test Configuration

- **Batch Size**: 2
- **Num Heads**: 8
- **Head Dim**: 64
- **Data Type**: float16
- **Iterations**: 10 per test

## Performance Comparison

| seq_len | Manual Attention | MLX SDPA | SageAttn MLX | vs Manual | vs SDPA |
|---------|-----------------|----------|--------------|-----------|---------|
| 256     | 1.12 ms         | 0.96 ms  | 0.96 ms      | 1.17x     | **1.00x** |
| 512     | 2.91 ms         | 2.32 ms  | 2.49 ms      | 1.17x     | 0.93x   |
| 1024    | 5.23 ms         | 2.90 ms  | 2.95 ms      | 1.78x     | 0.99x   |
| 2048    | 13.58 ms        | 7.05 ms  | 7.24 ms      | 1.88x     | 0.97x   |
| 4096    | 48.60 ms        | 27.19 ms | 27.47 ms     | 1.77x     | 0.99x   |
| 8192    | 198.40 ms       | 107.51 ms| 108.24 ms    | 1.83x     | 0.99x   |

### Legend

- **Manual Attention**: Standard attention implementation (matmul + softmax + matmul)
- **MLX SDPA**: `mx.fast.scaled_dot_product_attention` (MLX's fused Metal kernel)
- **SageAttn MLX**: `sageattn_mlx` with `smooth_k=True`
- **vs Manual**: Speedup of SageAttn over Manual (higher is better)
- **vs SDPA**: Ratio of SDPA time / SageAttn time (>1 means SageAttn is faster)

## Key Findings

### Performance

1. **SageAttention MLX achieves near-parity with MLX native SDPA** across all sequence lengths (0.93x-1.00x ratio).

2. **At seq_len=256**, SageAttention achieves **exact parity** with SDPA (1.00x), matching the fused Metal kernel performance.

3. **Compared to manual attention**, SageAttention provides **1.2x-1.9x speedup** across all sequence lengths.

### Accuracy

| Test | Max Diff | Mean Diff |
|------|----------|-----------|
| SageAttn (no smooth) vs Standard | 8.64e-7 | 3.70e-8 |
| SageAttn (smooth_k) vs Standard | 8.05e-7 | 3.71e-8 |

All differences are within floating-point tolerance, confirming numerical correctness.

## Implementation Notes

### Why INT8 Quantization Doesn't Help on MLX

Unlike NVIDIA's Tensor Cores that have native INT8 matrix multiplication hardware, Apple Silicon's GPU doesn't have equivalent hardware acceleration. When implementing INT8 quantization in software:

1. The quantization/dequantization overhead exceeds memory bandwidth savings
2. MLX must convert INT8 → float32 for matmul operations
3. `mx.fast.scaled_dot_product_attention` is already highly optimized

### What SageAttention MLX Does Instead

1. **K Mean Subtraction**: Reduces outliers in K, improving numerical stability
2. **Uses mx.fast.scaled_dot_product_attention**: Leverages MLX's fused Metal kernel
3. **LSE Correction**: Properly corrects log-sum-exp when smooth_k is enabled

This approach provides the algorithmic benefits of SageAttention (outlier reduction) while using the fastest available attention kernel on Apple Silicon.

## Running Benchmarks

```bash
# Full benchmark
PYTHONPATH=. python bench/benchmark_results.py

# With accuracy tests
PYTHONPATH=. python bench/benchmark_results.py --accuracy

# Quick benchmark only
PYTHONPATH=. python bench/benchmark.py --seq-len 1024 --iterations 10
```

## Comparison with NVIDIA SageAttention

| Feature | NVIDIA SageAttention | MLX SageAttention |
|---------|---------------------|-------------------|
| Q×K Quantization | INT8 (Tensor Cores) | Not used (no HW support) |
| K Smoothing | Mean subtraction | Mean subtraction ✓ |
| Attention Kernel | Triton/CUDA fused | mx.fast.sdpa (Metal) |
| P×V Precision | FP8/FP16 | FP16 |
| LSE Correction | Yes | Yes ✓ |

The MLX implementation matches the algorithmic approach of NVIDIA SageAttention while adapting to Apple Silicon's hardware capabilities.
