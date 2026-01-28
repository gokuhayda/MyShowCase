# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Dynamics Extensions (H-NCA)
===========================

Hyperbolic Neural Cellular Automata for emergent dynamics.

This module provides:
- HyperbolicNCA: Neural Cellular Automata on hyperbolic manifold (k-NN interface)
- HyperbolicNCAWrapper: H-NCA with adjacency matrix interface
- HNCAConfig: Configuration dataclass for H-NCA

The H-NCA implements local state dynamics on the Lorentz manifold,
following the update rule:
    h(t+1) = Exp_h(t)(-η * grad_h L_total)

This is the computational engine of the Ψ-SLM architecture.
"""

from .h_nca import HNCAConfig

try:
    from .h_nca import HyperbolicNCA, HyperbolicNCAWrapper
    __all__ = ["HNCAConfig", "HyperbolicNCA", "HyperbolicNCAWrapper"]
except ImportError:
    __all__ = ["HNCAConfig"]
