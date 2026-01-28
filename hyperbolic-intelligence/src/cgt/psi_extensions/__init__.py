# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Derived from PSI_SLM (MIT License) - Integrated into CGT with adaptations.

"""
PSI Extensions for CGT
======================

This module contains extensions derived from the Ψ-SLM project,
integrated into CGT following strict dominance rules.

IMPORTANT: This module is ADDITIVE ONLY. It does NOT modify any
existing CGT functionality. All CGT code remains intact.

Modules
-------
topology : Topological constraint fields and persistence landscapes
binding : H-AKOrN for solving the binding problem
dynamics : Hyperbolic Neural Cellular Automata
transfer : Gromov-Wasserstein and multi-teacher transfer
teachers : Teacher model wrappers (SentenceTransformer)
benchmarks : MTEB benchmark adapters
applications : Application pipelines (RAG)
visualization : Plotting utilities

Integration Notes
-----------------
- All modules use cgt.geometry.lorentz as the geometric substrate
- No circular dependencies with CGT core
- All imports are explicit
- This module is optional and can be removed without affecting CGT

Usage
-----
>>> from cgt.psi_extensions.topology import PersistenceLandscapeField
>>> from cgt.psi_extensions.binding import HyperbolicAKORN
>>> from cgt.psi_extensions.transfer import GromovWassersteinLoss

Author: Éric Gustavo Reis de Sena
Date: January 2026
Origin: Ψ-SLM Project (MIT License)
"""

__version__ = "0.1.0"
__origin__ = "psi_slm"
__integration_date__ = "2026-01-18"

# Lazy imports to avoid loading unnecessary modules
def __getattr__(name):
    """Lazy module loading for optional dependencies."""
    if name == "topology":
        from . import topology
        return topology
    elif name == "binding":
        from . import binding
        return binding
    elif name == "dynamics":
        from . import dynamics
        return dynamics
    elif name == "transfer":
        from . import transfer
        return transfer
    elif name == "teachers":
        from . import teachers
        return teachers
    elif name == "benchmarks":
        from . import benchmarks
        return benchmarks
    elif name == "applications":
        from . import applications
        return applications
    elif name == "visualization":
        from . import visualization
        return visualization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "topology",
    "binding", 
    "dynamics",
    "transfer",
    "teachers",
    "benchmarks",
    "applications",
    "visualization",
]
