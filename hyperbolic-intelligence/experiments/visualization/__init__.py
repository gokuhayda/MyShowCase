# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Visualization Extensions
========================

Plotting utilities, metrics logging, and H-AKORN visualization.

This module provides:
- MetricsLogger: Simple metrics tracking
- plot_poincare_embedding: Visualize in Poincaré disk
- plot_lorentz_embedding: Visualize on Lorentz hyperboloid (3D)
- plot_lorentz_2d_projection: Lorentz 2D projections
- plot_phase_coherence_matrix: Coherence heatmap
- plot_training_curves: Training loss curves
- H-AKORN: Hyperbolic Kuramoto oscillator visualization (dual mode)
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

# H-AKORN Simulator
from .hakorn_simulator import (
    HAKORNSimulatorConfig,
    HAKORNSimulator,
    LorentzGeometryGPU,
    run_hakorn_demo,
    plot_lorentz_3d,
    plot_poincare_2d,
    plot_evolution_triptych,
    plot_evolution_triptych_3d,
    plot_evolution_triptych_2d,
    plot_final_state,
    create_animation,
    create_animation_3d,
    create_animation_2d,
    create_hyperboloid_mesh,
    DEFAULT_CONCEPTS,
    CLUSTER_COLORS,
)

# H-AKORN Realtime with MTEB
from .hakorn_realtime import (
    RealtimeConfig,
    RealtimeHAKORN,
    MTEBDataLoader,
    MTEB_DATASETS,
    run_realtime_demo,
    visualize_sts_resonance,
)

# H-AKORN Video Recording
from .hakorn_video import (
    VideoConfig,
    FrameRenderer,
    HAKORNVideoRecorder,
    record_hakorn_video,
    record_mteb_video,
)

__all__ = [
    # Plotting
    "MetricsLogger",
    "plot_poincare_embedding",
    "plot_lorentz_embedding",
    "plot_lorentz_2d_projection",
    "plot_phase_coherence_matrix",
    "plot_training_curves",
    "plot_divergence_test_results",
    # H-AKORN Simulator
    "HAKORNSimulatorConfig",
    "HAKORNSimulator",
    "LorentzGeometryGPU",
    "run_hakorn_demo",
    "plot_lorentz_3d",
    "plot_poincare_2d",
    "plot_evolution_triptych",
    "plot_evolution_triptych_3d",
    "plot_evolution_triptych_2d",
    "plot_final_state",
    "create_animation",
    "create_animation_3d",
    "create_animation_2d",
    "create_hyperboloid_mesh",
    "DEFAULT_CONCEPTS",
    "CLUSTER_COLORS",
    # H-AKORN Realtime
    "RealtimeConfig",
    "RealtimeHAKORN",
    "MTEBDataLoader",
    "MTEB_DATASETS",
    "run_realtime_demo",
    "visualize_sts_resonance",
    # H-AKORN Video
    "VideoConfig",
    "FrameRenderer",
    "HAKORNVideoRecorder",
    "record_hakorn_video",
    "record_mteb_video",
]
