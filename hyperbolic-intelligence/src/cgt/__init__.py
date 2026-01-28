# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Contrastive Geometric Transfer (CGT)
====================================

Efficient sentence embeddings via hyperbolic distillation.

CGT achieves 24× compression (768→32 dimensions) while retaining
~93% of the teacher model's semantic similarity performance.

Package Structure
-----------------
cgt.geometry
    Lorentz (hyperbolic) manifold implementation.
cgt.losses
    Multi-objective loss functions.
cgt.regularization
    Lipschitz and stability regularization.
cgt.models
    Neural network architectures.
cgt.evaluation
    Metrics and falsification protocols.
cgt.experiments
    Training infrastructure.
cgt.utils
    General utilities.

Quick Start
-----------
>>> from cgt import CGTStudent, LorentzConfig
>>> from cgt.experiments import default_config, CGTTrainer
>>>
>>> # Create model
>>> config = default_config()
>>> student = CGTStudent(teacher_dim=384, student_dim=32)
>>>
>>> # Initialize homeostatic field
>>> student.init_homeostatic(n_anchors=16, alpha=0.1)

License
-------
CC BY-NC-SA 4.0 (Academic/Non-Commercial Use Only)
Commercial use is STRICTLY PROHIBITED without explicit permission.

Contact
-------
For commercial licensing: eirikreisena@gmail.com

Author
------
Éric Gustavo Reis de Sena

Date
----
January 2026
"""

__version__ = "1.0.0"
__author__ = "Éric Gustavo Reis de Sena"
__email__ = "eirikreisena@gmail.com"
__license__ = "CC BY-NC-SA 4.0"

# Copyright notice
_COPYRIGHT = """
Copyright © 2026 Éric Gustavo Reis de Sena.
All Rights Reserved.

This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0).

Patent Pending.
The methodology described herein may be subject to intellectual property protections.

For commercial licensing inquiries: eirikreisena@gmail.com
"""

# Core exports for convenience
from cgt.geometry import LorentzConfig, LorentzSubstrate
from cgt.losses import HyperbolicInfoNCE, MultiObjectiveLoss
from cgt.models import CGTStudent, HomeostaticField, RiemannianOptimizerWrapper, create_projector
from cgt.utils import academic_use_disclaimer, set_global_seed, clear_memory
from cgt.evaluation import (
    FalsificationProtocols,
    compute_effective_rank,
    compute_gromov_delta,
    compute_distortion,
    evaluate_stsb,
)
from cgt.experiments import (
    _EXPECTED_PARAMS, 
    validate_config_against_expected,
    CGTTrainer,
    ExperimentConfig,
    default_config,
    create_experiment,
    get_training_config,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Geometry
    "LorentzConfig",
    "LorentzSubstrate",
    # Losses
    "HyperbolicInfoNCE",
    "MultiObjectiveLoss",
    # Models
    "CGTStudent",
    "HomeostaticField",
    "RiemannianOptimizerWrapper",
    "create_projector",
    # Utils
    "set_global_seed",
    "academic_use_disclaimer",
    "clear_memory",
    # Evaluation
    "FalsificationProtocols",
    "compute_effective_rank",
    "compute_gromov_delta",
    "compute_distortion",
    "evaluate_stsb",
    # Experiments
    "_EXPECTED_PARAMS",
    "validate_config_against_expected",
    "CGTTrainer",
    "ExperimentConfig",
    "default_config",
    "create_experiment",
    "get_training_config",
]

# Show disclaimer on import (can be disabled)
_DISCLAIMER_SHOWN = False


def _show_disclaimer():
    """Show disclaimer once on first import."""
    global _DISCLAIMER_SHOWN
    if not _DISCLAIMER_SHOWN:
        print(_COPYRIGHT)
        _DISCLAIMER_SHOWN = True
