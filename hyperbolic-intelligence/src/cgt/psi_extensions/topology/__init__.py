# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Topology Extensions
===================

Topological constraint fields and persistence landscapes.

This module provides:
- PersistenceLandscapeField: Differentiable TDA proxy
- ScalarFeedbackField: Non-differentiable feedback for ablation
- TopologicalConstraintField: Abstract base class
- DifferentiableTopologicalLoss: Simplified wrapper for Ψ-SLM

Note: These complement (not replace) cgt.losses.core.TopoLoss
"""

from .topological_field import (
    TopologicalConfig,
    TopologicalConstraintField,
    PersistenceLandscapeField,
    ScalarFeedbackField,
    create_topological_field,
)

# DifferentiableTopologicalLoss requires torch
try:
    from .topological_field import DifferentiableTopologicalLoss
    __all__ = [
        "TopologicalConfig",
        "TopologicalConstraintField",
        "PersistenceLandscapeField",
        "ScalarFeedbackField",
        "create_topological_field",
        "DifferentiableTopologicalLoss",
    ]
except ImportError:
    __all__ = [
        "TopologicalConfig",
        "TopologicalConstraintField",
        "PersistenceLandscapeField",
        "ScalarFeedbackField",
        "create_topological_field",
    ]
