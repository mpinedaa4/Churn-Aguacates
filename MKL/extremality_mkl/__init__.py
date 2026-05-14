"""
extremality_mkl
===============
Librería para combinación de kernels usando ordenamiento por extremalidad.
"""

from .weights.extremality import kernel_extremality_weights, KernelWeights
from .kernels.polynomial import create_weak_kernels
from .metrics.kernel_metrics import (
    kernel_alignment,
    kernel_polarization,
    feature_space_measure,
    complex_ratio,
)

__all__ = [
    "kernel_extremality_weights",
    "KernelWeights",
    "create_weak_kernels",
    "kernel_alignment",
    "kernel_polarization",
    "feature_space_measure",
    "complex_ratio",
]