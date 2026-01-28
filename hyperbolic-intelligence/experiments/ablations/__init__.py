# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
CGT Ablation Studies
====================

Part IV experiments for isolating geometric contributions.

Modules:
- euclidean_ablation: Part IV.1 - CGT vs Euclidean baseline
- dimensional_ablation: Part IV.1b - Dimensional crossover analysis
- mrl_comparison: Part IV.2 - MRL truncation comparison
- bq_comparison: Part IV.3 - Binary Quantization comparison

AUDIT COMPLIANCE:
- All modules use ONLY hardened CGT modules for geometry
- Euclidean operations use ONLY PyTorch built-ins
- Binary operations use ONLY numpy (standard library)
- No formula derivation or notebook equation copying
"""

from .euclidean_ablation import (
    EuclideanStudent,
    EuclideanMultiObjectiveLoss,
    run_euclidean_ablation,
    AblationConfig,
)
from .dimensional_ablation import (
    DimensionalAblationConfig,
    run_dimensional_ablation,
)
from .mrl_comparison import (
    MRLTruncation,
    MRLConfig,
    run_mrl_comparison,
)
from .bq_comparison import (
    BinaryQuantization,
    BQComparisonConfig,
    run_bq_comparison,
)

__all__ = [
    # Part IV.1 - Euclidean Ablation
    "EuclideanStudent",
    "EuclideanMultiObjectiveLoss",
    "run_euclidean_ablation",
    "AblationConfig",
    # Part IV.1b - Dimensional Ablation
    "DimensionalAblationConfig",
    "run_dimensional_ablation",
    # Part IV.2 - MRL Comparison
    "MRLTruncation",
    "MRLConfig",
    "run_mrl_comparison",
    # Part IV.3 - BQ Comparison
    "BinaryQuantization",
    "BQComparisonConfig",
    "run_bq_comparison",
]
