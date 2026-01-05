"""
Setup script for SageAttention MLX port.
"""

from setuptools import setup, find_packages

setup(
    name="sageattention-mlx",
    version="0.1.0",
    description="Apple Silicon optimized quantized attention for MLX",
    author="SageAttention Team",
    license="Apache 2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mlx>=0.0.1",
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.990",
        ],
        "examples": [
            "transformers>=4.30",
            "datasets>=2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
