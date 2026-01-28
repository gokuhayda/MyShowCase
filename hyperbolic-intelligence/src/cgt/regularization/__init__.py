# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Regularization Module
=========================

Regularization techniques for geometric stability.

Exports
-------
LipschitzRegularizer
    Lipschitz continuity regularization.
SpectralNormRegularizer
    Spectral norm monitoring.
LipschitzAnalyzer
    Comprehensive Lipschitz analysis (local, global, PAC).
LipschitzReport
    Analysis report dataclass.
"""

from cgt.regularization.lipschitz import (
    LipschitzRegularizer,
    SpectralNormRegularizer,
)
from cgt.regularization.lipschitz_analysis import (
    LipschitzAnalyzer,
    LipschitzReport,
    LipschitzRegularizer as LipschitzRegularizerV2,
    SpectralNormRegularizer as SpectralNormRegularizerV2,
)

__all__ = [
    "LipschitzRegularizer",
    "SpectralNormRegularizer",
    "LipschitzAnalyzer",
    "LipschitzReport",
]
