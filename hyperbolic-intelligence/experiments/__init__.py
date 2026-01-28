# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
CGT Experiments
===============

Complete experimental suite for CGT paper validation.

AUDIT COMPLIANCE:
- All experiments use ONLY hardened CGT modules
- No formula derivation or geometry implementation
- Fair baselines using PyTorch built-ins only

Submodules:
- ablations: Part IV experiments (Euclidean, Dimensional, MRL, BQ comparisons)
- benchmarks: Part I.19, II, IV.4 (compression, multi-model, latency)
- analysis: Part VI (statistical robustness)
"""

from .ablations import (
    # Part IV.1 - Euclidean Ablation
    EuclideanStudent,
    EuclideanMultiObjectiveLoss,
    run_euclidean_ablation,
    AblationConfig,
    # Part IV.1b - Dimensional Ablation
    DimensionalAblationConfig,
    run_dimensional_ablation,
    # Part IV.2 - MRL Comparison
    MRLTruncation,
    MRLConfig,
    run_mrl_comparison,
    # Part IV.3 - BQ Comparison
    BinaryQuantization,
    BQComparisonConfig,
    run_bq_comparison,
)

from .benchmarks import (
    # Part I.19 - Cascade Compression
    ScalarQuantizer,
    BinaryQuantizer,
    ProductQuantizer,
    run_cascade_compression,
    # Part II - Multi-Model Benchmark
    MultiModelConfig,
    PCABaseline,
    run_multi_model_benchmark,
    # Part IV.4 - Latency Benchmark
    LatencyConfig,
    run_latency_benchmark,
)

from .analysis import (
    # Part VI - Statistical Robustness
    RobustnessConfig,
    run_statistical_robustness,
)

__all__ = [
    # Ablations
    "EuclideanStudent",
    "EuclideanMultiObjectiveLoss",
    "run_euclidean_ablation",
    "AblationConfig",
    "DimensionalAblationConfig",
    "run_dimensional_ablation",
    "MRLTruncation",
    "MRLConfig",
    "run_mrl_comparison",
    "BinaryQuantization",
    "BQComparisonConfig",
    "run_bq_comparison",
    # Benchmarks
    "ScalarQuantizer",
    "BinaryQuantizer",
    "ProductQuantizer",
    "run_cascade_compression",
    "MultiModelConfig",
    "PCABaseline",
    "run_multi_model_benchmark",
    "LatencyConfig",
    "run_latency_benchmark",
    # Analysis
    "RobustnessConfig",
    "run_statistical_robustness",
]
