# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Visualization Extensions
========================

Plotting utilities and metrics logging.

This module provides:
- MetricsLogger: Simple metrics tracking
- plot_poincare_embedding: Visualize in Poincaré disk
- plot_lorentz_embedding: Visualize on Lorentz hyperboloid (3D)
- plot_lorentz_2d_projection: Lorentz 2D projections
- plot_phase_coherence_matrix: Coherence heatmap
- plot_training_curves: Training loss curves
"""

from .plotting import (
    MetricsLogger,
    plot_poincare_embedding,
    plot_lorentz_embedding,
    plot_lorentz_2d_projection,
    plot_phase_coherence_matrix,
    plot_training_curves,
    plot_divergence_test_results,
)

__all__ = [
    "MetricsLogger",
    "plot_poincare_embedding",
    "plot_lorentz_embedding",
    "plot_lorentz_2d_projection",
    "plot_phase_coherence_matrix",
    "plot_training_curves",
    "plot_divergence_test_results",
]
