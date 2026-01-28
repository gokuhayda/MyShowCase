# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Experiments Module
======================

Experiment configuration and training infrastructure.

HARDENED versions are the production-ready implementations matching
the CGT_Paper_Ready_v6_1_HARDENED notebook exactly.

Exports
-------
ExperimentConfig
    Canonical hyperparameter configuration.
default_config
    Factory for default configuration.
create_experiment
    Complete experiment setup factory.
get_training_config
    Convert config to trainer format.
CGTTrainer
    Training orchestrator (V9.9.3 - Float64).
RiemannianOptimizerWrapper
    Optimizer wrapper for manifold parameters.
_EXPECTED_PARAMS
    Validation reference for Part III.
validate_config_against_expected
    Validate configuration against expected values.
"""

from cgt.experiments.part1_reference import (
    ExperimentConfig,
    _EXPECTED_PARAMS,
    create_experiment,
    default_config,
    get_training_config,
    validate_config_against_expected,
)

# HARDENED trainer (production)
from cgt.experiments.trainer_hardened import (
    CGTTrainer,
    RiemannianOptimizerWrapper,
)

__all__ = [
    "ExperimentConfig",
    "_EXPECTED_PARAMS",
    "default_config",
    "create_experiment",
    "get_training_config",
    "validate_config_against_expected",
    "CGTTrainer",
    "RiemannianOptimizerWrapper",
]
