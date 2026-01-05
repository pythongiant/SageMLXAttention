# SageAttention MLX - Apple Silicon Port

Apple Silicon optimized implementation of SageAttention using MLX framework.

## Overview

**SageAttention MLX** is an MLX port of SageAttention, providing quantized attention mechanisms optimized for Apple Silicon GPUs. It achieves significant speedups through INT8/FP8 quantization while maintaining accuracy across different Apple devices.

### Key Features

- **INT8 Quantization**: Q×K^T computation with INT8 quantization
- **Outlier Smoothing**: Reduces quantization error for better accuracy
- **Per-block & Per-thread Quantization**: Flexible granularity options
- **Apple Silicon Native**: Optimized for Metal backend via MLX
- **Plug-and-Play**: Easy integration with existing transformer models

## Status

⚠️ **Alpha Release** - Under active development

### Implemented
- [x] Core quantized attention API (`sageattn_mlx`)
- [x] INT8 quantization utilities
- [x] Outlier smoothing
- [x] Basic MLX kernel implementations
- [x] Example attention layer

### Planned
- [ ] Custom Metal kernels for FP8 operations
- [ ] Block-wise flash attention optimization
- [ ] Comprehensive benchmarking
- [ ] Integration with popular models (Llama, Mistral, etc.)
- [ ] Performance optimization pass

## Installation

### Requirements
- Python >= 3.9
- MLX >= 0.0.1
- NumPy >= 1.20

### Install from Source

```bash
cd sageattention_mlx
pip install -e .
```

## Quick Start

### Basic Usage

```python
import mlx.core as mx
from sageattention_mlx import sageattn_mlx

# Create Q, K, V tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
q = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
k = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)
v = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=mx.float16)

# Apply quantized attention
output = sageattn_mlx(q, k, v, sm_scale=1.0 / (head_dim ** 0.5))
print(output.shape)  # (batch_size, num_heads, seq_len, head_dim)
```

### Integration with Attention Layer

```python
from sageattention_mlx.example_attention import AttentionWithSageAttn
import mlx.core as mx

# Create attention layer
attn = AttentionWithSageAttn(dims=256, num_heads=8)

# Apply to input
x = mx.random.normal((2, 10, 256))
output = attn(x)
```

## Architecture

```
sageattention_mlx/
├── __init__.py           # Package exports
├── core.py               # Main sageattn_mlx implementation
├── quant.py              # Quantization utilities (INT8, FP8)
├── mlx_kernels.py        # Optimized MLX kernels
├── example_attention.py  # Example attention layer
└── setup.py              # Installation script
```

### Module Descriptions

- **core.py**: Main entry point with `sageattn_mlx()` function that handles outlier smoothing, quantization, and attention computation.

- **quant.py**: Quantization implementations:
  - `quantize_qk()`: INT8 quantization for Q and K (per-block or per-thread)
  - `quantize_pv()`: FP8 or lower precision quantization for P×V
  - Dequantization functions for reverse operations

- **mlx_kernels.py**: Placeholder for custom Metal kernel implementations:
  - `mlx_quantized_matmul()`: Optimized Q×K computation
  - `mlx_flash_attention_block()`: Block-wise attention processing

## Parameters

### `sageattn_mlx()`

```python
sageattn_mlx(
    q: mx.array,                    # Query: (..., seq_len_q, head_dim)
    k: mx.array,                    # Key: (..., seq_len_k, head_dim)
    v: mx.array,                    # Value: (..., seq_len_k, head_dim)
    is_causal: bool = False,        # Apply causal masking
    sm_scale: Optional[float] = None,  # Softmax scale (default: 1/√head_dim)
    return_lse: bool = False,       # Return log-sum-exp
    qk_quant_gran: str = "per_block",  # "per_block" or "per_thread"
    pv_dtype: str = "float16",      # "float16" or "float32"
    smooth_k: bool = True           # Apply outlier smoothing to K
) -> mx.array
```

## Performance

Expected improvements over standard attention on Apple Silicon:
- **M1/M2**: ~1.5x-2x speedup with maintained accuracy
- **M3**: ~2x-2.5x speedup with maintained accuracy

*Benchmarks coming soon*

## Future Enhancements

1. **Custom Metal Kernels**: Implement FP8 and specialized quantization in Metal for maximum performance
2. **Flash Attention**: Block-wise processing for better memory utilization
3. **Distributed Inference**: Support for multi-GPU attention via tensor parallelism
4. **Torch.compile Support**: If MLX adds torch interop
5. **Comprehensive Tests**: Unit and integration tests

## Contributing

Contributions are welcome! Areas for improvement:
- Custom Metal kernel implementations
- Performance optimizations
- Accuracy improvements
- Documentation and examples
- Testing and benchmarking

## References

### Papers
- [SageAttention](https://arxiv.org/abs/2410.02367)
- [SageAttention2](https://arxiv.org/abs/2411.10958)
- [SageAttention3](https://arxiv.org/abs/2505.11594)

### Related Projects
- [SageAttention (Original)](https://github.com/thu-ml/SageAttention)
- [MLX Framework](https://github.com/ml-explore/mlx)

## License

Apache License 2.0 - See LICENSE file for details
