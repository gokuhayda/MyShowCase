# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Binding Extensions (H-AKOrN)
============================

Hyperbolic Artificial Kuramoto Oscillatory Neurons for solving
the binding problem through phase synchronization.

This module provides:
- HyperbolicAKORN: Full phase dynamics on hyperbolic manifold
- HAKORNLayer: Simplified layer for model integration
- HAKORNConfig: Configuration dataclass
- PhaseCoherenceLoss: Loss for encouraging synchronization
"""

from .h_akorn import HAKORNConfig

try:
    from .h_akorn import (
        HyperbolicAKORN,
        HAKORNLayer,
        PhaseCoherenceLoss,
    )
    __all__ = [
        "HAKORNConfig",
        "HyperbolicAKORN",
        "HAKORNLayer",
        "PhaseCoherenceLoss",
    ]
except ImportError:
    __all__ = ["HAKORNConfig"]
