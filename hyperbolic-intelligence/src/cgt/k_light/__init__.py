# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
K_Light Module (STS-specific)
=============================

Minimal extension for K_Lighting STS/ranking encoder.
Reuses CGT infrastructure for geometry and training.

This module is OPTIONAL and can be removed without affecting CGT core.
"""

from cgt.k_light.losses import FormanRicciLoss, CoherenceLoss, KLightMultiObjectiveLoss

__all__ = [
    "FormanRicciLoss",
    "CoherenceLoss", 
    "KLightMultiObjectiveLoss",
]
