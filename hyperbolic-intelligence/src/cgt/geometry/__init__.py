# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Geometry Module
===================

Provides geometric substrates for hyperbolic neural networks.

HARDENED versions are the production-ready implementations matching
the CGT_Paper_Ready_v6_1_HARDENED notebook exactly.

Exports
-------
LorentzConfig
    Configuration dataclass for Lorentz manifold.
LorentzSubstrateHardened
    Production-ready Lorentz hyperboloid implementation.
LorentzFrechetMean
    Exact Fréchet mean solver (Lou et al. 2020).
EinsteinMidpoint
    Fast approximation for hyperbolic midpoint.
frechet_mean
    Convenience function for Fréchet mean computation.
safe_acosh
    Numerically stable inverse hyperbolic cosine.
safe_sqrt
    Numerically stable square root.
"""

# HARDENED versions (production)
from cgt.geometry.lorentz_hardened import (
    LorentzConfig,
    LorentzSubstrateHardened,
    safe_acosh,
    safe_sqrt,
)

# Alias for backward compatibility
LorentzSubstrate = LorentzSubstrateHardened

# Fréchet mean (advanced)
from cgt.geometry.frechet import (
    LorentzFrechetMean,
    EinsteinMidpoint,
    frechet_mean,
)

__all__ = [
    # Core (HARDENED)
    "LorentzConfig",
    "LorentzSubstrateHardened",
    "LorentzSubstrate",  # Alias
    # Fréchet mean
    "LorentzFrechetMean",
    "EinsteinMidpoint",
    "frechet_mean",
    # Utilities
    "safe_acosh",
    "safe_sqrt",
]
