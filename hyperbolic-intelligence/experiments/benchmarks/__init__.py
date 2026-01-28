# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
CGT Benchmarks Module
=====================

Performance benchmarks for CGT experiments.

Modules:
- cascade_compression: Part I.19 - Post-CGT quantization analysis
- multi_model_benchmark: Part II - Multi-model generalization
- latency_benchmark: Part IV.4 - Similarity computation timing
"""

from .cascade_compression import (
    ScalarQuantizer,
    BinaryQuantizer,
    ProductQuantizer,
    run_cascade_compression,
)
from .multi_model_benchmark import (
    MultiModelConfig,
    PCABaseline,
    run_multi_model_benchmark,
)
from .latency_benchmark import (
    LatencyConfig,
    run_latency_benchmark,
)

__all__ = [
    # Part I.19 - Cascade Compression
    "ScalarQuantizer",
    "BinaryQuantizer",
    "ProductQuantizer",
    "run_cascade_compression",
    # Part II - Multi-Model Benchmark
    "MultiModelConfig",
    "PCABaseline",
    "run_multi_model_benchmark",
    # Part IV.4 - Latency Benchmark
    "LatencyConfig",
    "run_latency_benchmark",
]
