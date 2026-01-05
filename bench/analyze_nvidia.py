"""
Analysis of NVIDIA SageAttention implementation vs our MLX version.
Focuses on quantization approach and scaling factors.
"""

import mlx.core as mx
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sageattention_mlx import quantize_qk


def analyze_quantization_approach():
    """Analyze and compare quantization methods."""
    
    print("=" * 90)
    print("QUANTIZATION APPROACH ANALYSIS")
    print("=" * 90)
    
    # Create test data
    x = mx.random.normal((2, 8, 256, 64), dtype=mx.float32)
    
    # NVIDIA approach parameters
    sm_scale = 1.0 / math.sqrt(64)  # 1/sqrt(D) where D=64
    exp2_factor = 1.44269504  # ln(2)
    
    print(f"\nInput Statistics:")
    print(f"  Shape: {x.shape}")
    print(f"  Mean: {mx.mean(x):.6f}")
    print(f"  Std: {mx.std(x):.6f}")
    print(f"  Min: {mx.min(x):.6f}")
    print(f"  Max: {mx.max(x):.6f}")
    print(f"  Abs Max: {mx.max(mx.abs(x)):.6f}")
    
    print(f"\nNVIDIA Quantization Parameters:")
    print(f"  sm_scale: {sm_scale:.8f}")
    print(f"  exp2_factor: {exp2_factor:.8f}")
    print(f"  Q baking: sm_scale * exp2_factor = {sm_scale * exp2_factor:.8f}")
    
    print(f"\n--- Test 1: Q Quantization (is_q=True) ---")
    q_int8, q_scales, q_zp = quantize_qk(
        x, gran="per_block", block_size=128, 
        sm_scale=sm_scale, is_q=True
    )
    
    print(f"  Q Int8 shape: {q_int8.shape}")
    print(f"  Q scales shape: {q_scales.shape}")
    print(f"  Q scales (first 3): {q_scales[0, 0, :3]}")
    print(f"  Q zero_points (first 3): {q_zp[0, 0, :3]}")
    print(f"  Expected scale ~= {mx.max(mx.abs(x * (sm_scale * exp2_factor))) / 127:.6f}")
    
    print(f"\n--- Test 2: K Quantization (is_q=False) ---")
    k_int8, k_scales, k_zp = quantize_qk(
        x, gran="per_block", block_size=128,
        sm_scale=1.0, is_q=False  # K uses sm_scale=1.0
    )
    
    print(f"  K Int8 shape: {k_int8.shape}")
    print(f"  K scales shape: {k_scales.shape}")
    print(f"  K scales (first 3): {k_scales[0, 0, :3]}")
    print(f"  K zero_points (first 3): {k_zp[0, 0, :3]}")
    print(f"  Expected scale ~= {mx.max(mx.abs(x)) / 127:.6f}")
    
    print(f"\n--- Verification: Scale Relationships ---")
    scale_ratio = q_scales[0, 0, 0] / k_scales[0, 0, 0]
    expected_ratio = sm_scale * exp2_factor / 1.0
    print(f"  Q scale / K scale: {scale_ratio:.6f}")
    print(f"  Expected ratio (sm_scale * exp2): {expected_ratio:.6f}")
    print(f"  Match: {'✅' if abs(scale_ratio - expected_ratio) < expected_ratio * 0.1 else '❌'}")
    

def analyze_nvidia_reference():
    """Detailed analysis of NVIDIA's implementation."""
    
    print("\n\n")
    print("=" * 90)
    print("NVIDIA SAGEATTENTION REFERENCE ANALYSIS")
    print("=" * 90)
    


def profile_scale_computation():
    """Profile the scale computation for different block sizes."""
    
    print("\n\n")
    print("=" * 90)
    print("SCALE COMPUTATION PROFILING")
    print("=" * 90)
    
    # Create test data
    x = mx.random.normal((2, 8, 2048, 64), dtype=mx.float32)
    
    block_sizes = [32, 64, 128, 256, 512]
    sm_scale = 1.0 / math.sqrt(64)
    
    print(f"\nInput: shape={x.shape}, will test quantization with different block sizes")
    
    for block_size in block_sizes:
        try:
            q_int8, q_scales, q_zp = quantize_qk(
                x, gran="per_block", block_size=block_size,
                sm_scale=sm_scale, is_q=True
            )
            
            n_blocks = q_scales.shape[-1]
            scale_mean = mx.mean(q_scales)
            scale_std = mx.std(q_scales)
            
            print(f"  Block size {block_size:3d}: "
                  f"n_blocks={n_blocks:4d}, "
                  f"scale_mean={scale_mean:.6f}, "
                  f"scale_std={scale_std:.6f}")
        except Exception as e:
            print(f"  Block size {block_size:3d}: ERROR - {str(e)[:50]}")


if __name__ == "__main__":
    analyze_quantization_approach()
    analyze_nvidia_reference()
    profile_scale_computation()
    
    print("\n\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
