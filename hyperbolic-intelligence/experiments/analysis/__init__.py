# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
CGT Analysis Module
===================

Statistical analysis for CGT experiments.

Modules:
- statistical_robustness: Part VI - Multi-seed sensitivity analysis
- storage_efficiency: Part VIII - Storage efficiency analysis
"""

from .statistical_robustness import (
    RobustnessConfig,
    run_statistical_robustness,
)
from .storage_efficiency import (
    run_storage_analysis,
    plot_storage_efficiency,
)

__all__ = [
    # Part VI - Statistical Robustness
    "RobustnessConfig",
    "run_statistical_robustness",
    # Part VIII - Storage Efficiency
    "run_storage_analysis",
    "plot_storage_efficiency",
]
