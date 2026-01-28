# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Distillation Module
==================

Knowledge distillation from Euclidean teachers to Hyperbolic models.
"""

from .gpt2_distillation import (
    DistillationConfig,
    GPT2TeacherWrapper,
    DistillationTrainer,
    plot_distillation_analysis,
)

__all__ = [
    'DistillationConfig',
    'GPT2TeacherWrapper', 
    'DistillationTrainer',
    'plot_distillation_analysis',
]
