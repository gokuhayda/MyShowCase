# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Loss Functions Module
=========================

Multi-objective loss functions for Contrastive Geometric Transfer.

HARDENED versions are the production-ready implementations matching
the CGT_Paper_Ready_v6_1_HARDENED notebook exactly.

Exports
-------
HyperbolicInfoNCE_Lorentz
    Contrastive loss using geodesic distance matrix.
PowerLawDistillation
    Distance compression d_target = d_teacher^α.
KLDistillation
    KL-divergence distillation for gradient stability. (v9.9.4)
TopoLoss
    Topological regularization with temperature scheduling.
LipschitzRegularizer
    Metric smoothness penalty (ratio formulation).
SpectralManifoldAlignmentHardened
    Spectral alignment with numerical stability.
MultiObjectiveLoss
    Combined multi-objective loss (V9.9.4 with KL option).

Hyperbolic LLM Losses
---------------------
HyperbolicLMLoss
    Combined loss for hyperbolic language modeling.
HyperbolicLLMTrainingLoss
    Complete training loss with teacher distillation.
TeacherDistillationLoss
    Teacher-student distillation for hyperbolic models.
RadiusRegularization
    Soft radius constraint to prevent boundary drift.
ManifoldFidelityLoss
    Manifold constraint violation penalty.
"""

# HARDENED versions (production)
from cgt.losses.losses_hardened import (
    HyperbolicInfoNCE_Lorentz,
    PowerLawDistillation,
    KLDistillation,  # AUDIT FIX v9.9.4
    TopoLoss,
    LipschitzRegularizer,
    SpectralManifoldAlignmentHardened,
    SpectralManifoldAlignment,
    MultiObjectiveLoss,
    MultiObjectiveLossHardened,
)

# Hyperbolic LLM Losses
from cgt.losses.hyperbolic_lm_losses import (
    HyperbolicLMLoss,
    HyperbolicLLMTrainingLoss,
    TeacherDistillationLoss,
    RadiusRegularization,
    ManifoldFidelityLoss,
    HyperbolicInfoNCE as HyperbolicInfoNCE_LLM,
)

# Alias for backward compatibility
HyperbolicInfoNCE = HyperbolicInfoNCE_Lorentz

# Bootstrap (advanced) - optional
try:
    from cgt.losses.topo_bootstrap import (
        BootstrapTopoLoss,
        AdaptiveTopoLoss,
        SpectralBettiEstimator,
        BootstrapBettiEstimator,
        BettiEstimate,
    )
    _BOOTSTRAP_AVAILABLE = True
except ImportError:
    _BOOTSTRAP_AVAILABLE = False

__all__ = [
    # Core (HARDENED)
    "HyperbolicInfoNCE_Lorentz",
    "HyperbolicInfoNCE",  # Alias
    "PowerLawDistillation",
    "KLDistillation",  # AUDIT FIX v9.9.4
    "TopoLoss",
    "LipschitzRegularizer",
    "SpectralManifoldAlignmentHardened",
    "SpectralManifoldAlignment",
    "MultiObjectiveLoss",
    "MultiObjectiveLossHardened",
    # Hyperbolic LLM Losses
    "HyperbolicLMLoss",
    "HyperbolicLLMTrainingLoss",
    "TeacherDistillationLoss",
    "RadiusRegularization",
    "ManifoldFidelityLoss",
    "HyperbolicInfoNCE_LLM",
    # Hybrid Active Loss
    "HybridActiveLoss",
    "HybridLossConfig",
    "create_hybrid_loss",
]

# Import Hybrid Active Loss
try:
    from cgt.losses.hybrid_active_loss import (
        HybridActiveLoss,
        HybridLossConfig,
        create_hybrid_loss,
    )
    _HYBRID_AVAILABLE = True
except ImportError as e:
    _HYBRID_AVAILABLE = False
    import warnings
    warnings.warn(f"HybridActiveLoss not available: {e}")

# Add bootstrap exports if available
if _BOOTSTRAP_AVAILABLE:
    __all__.extend([
        "BootstrapTopoLoss",
        "AdaptiveTopoLoss",
        "SpectralBettiEstimator",
        "BootstrapBettiEstimator",
        "BettiEstimate",
    ])
