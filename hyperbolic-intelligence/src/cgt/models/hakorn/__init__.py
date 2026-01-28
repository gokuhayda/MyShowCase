# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
H-AKORN: Hyperbolic Attention with Kuramoto Oscillator Regularized Networks
===========================================================================

Implementation of hyperbolic transformer with Kuramoto oscillator dynamics
for phase synchronization and adaptive coupling.

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

__version__ = "0.1.0"
__author__ = "Éric Gustavo Reis de Sena"

from .phase_dynamics import KuramotoPhaseEvolution
from .coupling import AdaptiveCoupling
from .attention import HyperbolicKuramotoAttention
from .layer import HAKORNLayer
from .model import HAKORNTransformer
from .losses import HAKORNLoss

__all__ = [
    "KuramotoPhaseEvolution",
    "AdaptiveCoupling",
    "HyperbolicKuramotoAttention",
    "HAKORNLayer",
    "HAKORNTransformer",
    "HAKORNLoss",
]
