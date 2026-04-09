# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Transfer Learning Extensions
============================

Gromov-Wasserstein and multi-teacher transfer learning.

Note: Requires `POT` library (pip install POT).
If POT or TensorFlow is unavailable (e.g. Colab GPU runtime),
imports degrade gracefully — GW features become unavailable
but the rest of the cgt package loads normally.
"""

try:
    from .gw_transfer import (
        compute_euclidean_cost_matrix,
        compute_hyperbolic_cost_matrix,
        entropic_gromov_wasserstein_loss,
        GromovWassersteinLoss,
        compute_gw_divergence,
        GeometricTransferTrainer,
    )
    _GW_AVAILABLE = True
except (ImportError, OSError):
    # POT not installed or TF backend broken (common on Colab GPU runtimes)
    compute_euclidean_cost_matrix   = None
    compute_hyperbolic_cost_matrix  = None
    entropic_gromov_wasserstein_loss = None
    GromovWassersteinLoss           = None
    compute_gw_divergence           = None
    GeometricTransferTrainer        = None
    _GW_AVAILABLE = False

__all__ = [
    "compute_euclidean_cost_matrix",
    "compute_hyperbolic_cost_matrix",
    "entropic_gromov_wasserstein_loss",
    "GromovWassersteinLoss",
    "compute_gw_divergence",
    "GeometricTransferTrainer",
]
