# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Transfer Learning Extensions
============================

Gromov-Wasserstein and multi-teacher transfer learning.

This module provides:
- GromovWassersteinLoss: Alternative to PowerLawDistillation
- entropic_gromov_wasserstein_loss: Functional interface
- compute_gw_divergence: Divergence between configurations
- GeometricTransferTrainer: Training coordinator

Note: Requires `POT` library (pip install POT)
"""

from .gw_transfer import (
    compute_euclidean_cost_matrix,
    compute_hyperbolic_cost_matrix,
    entropic_gromov_wasserstein_loss,
    GromovWassersteinLoss,
    compute_gw_divergence,
    GeometricTransferTrainer,
)

__all__ = [
    "compute_euclidean_cost_matrix",
    "compute_hyperbolic_cost_matrix",
    "entropic_gromov_wasserstein_loss",
    "GromovWassersteinLoss",
    "compute_gw_divergence",
    "GeometricTransferTrainer",
]
