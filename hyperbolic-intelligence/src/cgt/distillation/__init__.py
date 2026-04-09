# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Distillation Module
==================

Knowledge distillation from Euclidean teachers to Hyperbolic models.
"""

from .distillation_v2 import (
    DistillationConfigV2,
    GPT2TeacherWrapperV2,
    DistillationTrainerV2,
    TeacherDistillationLossV2,
)


from .dataset_v2 import (
    WikiTextTokenDataset,
    build_wikitext_loaders,
)

from .gpt2_distillation import (
    DistillationConfig,
    GPT2TeacherWrapper,
    DistillationTrainer,
    plot_distillation_analysis,
)


from .hyperbolic_projector import (
    HiddenProjector,
    HyperbolicProjectorV2,
    HyperbolicProjectorV3,
)

__all__ = [
    'DistillationConfigV2',
    'GPT2TeacherWrapperV2',
    'DistillationTrainerV2',
    'TeacherDistillationLossV2',
    'HiddenProjector',
    'HyperbolicProjectorV2',
    'HyperbolicProjectorV3',

    'WikiTextTokenDataset',
    'build_wikitext_loaders',
    'DistillationConfig',
    'GPT2TeacherWrapper', 
    'DistillationTrainer',
    'plot_distillation_analysis',
]

# Paper 2 — structural DegEq prevention
from .geometric_distillation import (
    ProjectiveKLLoss,
    DecoupledRadialAngularLoss,
    compute_f2,
)
