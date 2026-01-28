# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part I Reference Experiment
===========================

SINGLE SOURCE OF TRUTH for CGT experiment configuration and execution.

This module defines the canonical experiment setup that all other
experiments (Part II, III, IV) must import from. No duplicated logic.

Configuration
-------------
All hyperparameters are defined here and imported elsewhere.

Functions
---------
- create_experiment: Factory for experiment setup
- default_config: Returns canonical hyperparameter dictionary

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

# ═══════════════════════════════════════════════════════════════════════════════
#                    CANONICAL HYPERPARAMETER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED PARAMETERS - Validation Reference for Part III
# ═══════════════════════════════════════════════════════════════════════════════
# These are the expected default values that Part III should validate against.
# If Part III receives parameters that differ significantly from these,
# it should log a warning to ensure intentional deviation.

_EXPECTED_PARAMS = {
    'LAMBDA_CONTRASTIVE': 1.0,
    'LAMBDA_DISTILL': 0.5,
    'LAMBDA_SPECTRAL': 0.1,
    'LAMBDA_TOPO': 0.05,
    'LAMBDA_LIPSCHITZ': 0.01,
    'N_ANCHORS': 16,
    'HOMEOSTATIC_ALPHA': 0.1,
    'TEMPERATURE': 0.07,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS': 50,
    'PATIENCE': 10
}


def validate_config_against_expected(config: "ExperimentConfig") -> dict:
    """
    Validate configuration against expected Part III parameters.

    Args:
        config: ExperimentConfig to validate.

    Returns:
        Dictionary with deviations from expected values.

    Notes:
        - Returns empty dict if all values match
        - Logs warning for each deviation
        - Does NOT prevent execution, only warns
    """
    import warnings

    deviations = {}
    config_mapping = {
        'LAMBDA_CONTRASTIVE': config.lambda_contrastive,
        'LAMBDA_DISTILL': config.lambda_distill,
        'LAMBDA_SPECTRAL': config.lambda_spectral,
        'LAMBDA_TOPO': config.lambda_topo,
        'LAMBDA_LIPSCHITZ': config.lambda_lipschitz,
        'N_ANCHORS': config.n_anchors,
        'HOMEOSTATIC_ALPHA': config.homeostatic_alpha,
        'TEMPERATURE': config.temperature,
        'BATCH_SIZE': config.batch_size,
        'LEARNING_RATE': config.learning_rate,
        'NUM_EPOCHS': config.num_epochs,
        'PATIENCE': config.patience,
    }

    for param, expected in _EXPECTED_PARAMS.items():
        actual = config_mapping.get(param)
        if actual is not None and actual != expected:
            deviations[param] = {'expected': expected, 'actual': actual}
            warnings.warn(
                f"Config deviation: {param} = {actual} (expected {expected})",
                UserWarning,
                stacklevel=2
            )

    return deviations


@dataclass
class ExperimentConfig:
    """
    Canonical experiment configuration.

    All experiments MUST use this configuration or explicitly
    document deviations with justification.

    Notes:
        - This is the SINGLE SOURCE OF TRUTH
        - Part II/III/IV import from here
        - Changes here propagate to all experiments
    """

    # Model Architecture
    teacher_model: str = "all-MiniLM-L6-v2"
    hyperbolic_dim: int = 32
    hidden_dim: int = 256

    # Training Parameters
    batch_size: int = 64
    learning_rate: float = 1e-4  # Updated to match _EXPECTED_PARAMS
    num_epochs: int = 50  # Updated to match _EXPECTED_PARAMS
    patience: int = 10  # Updated to match _EXPECTED_PARAMS
    weight_decay: float = 0.01

    # Loss Weights (HGST) - Updated to match _EXPECTED_PARAMS
    lambda_contrastive: float = 1.0
    lambda_distill: float = 0.5  # Updated from 0.7
    lambda_spectral: float = 0.1  # Updated from 0.2
    lambda_topo: float = 0.05  # Updated from 0.02
    lambda_lipschitz: float = 0.01  # Updated from 0.005
    radius_weight: float = 0.05

    # Contrastive
    temperature: float = 0.07
    target_beta_0: float = 1.0
    power_law_alpha: float = 0.5

    # Spectral/Lipschitz
    use_spectral_norm: bool = True
    lipschitz_noise_scale: float = 0.05

    # Homeostatic Field
    enable_homeostatic: bool = True
    n_anchors: int = 16
    homeostatic_alpha: float = 0.1

    # Lorentz Substrate
    initial_curvature: float = 1.0
    learnable_curvature: bool = True
    curvature_min: float = 0.1  # HARDENED: was 0.5
    curvature_max: float = 10.0  # HARDENED: was 2.0

    # Radius Regularization
    lambda_radius: float = 0.05
    target_radius: float = 2.0
    radius_bound: float = 10.0

    # Falsification Protocols
    enable_f1_homotopy: bool = True
    enable_f2_stability: bool = True
    enable_f3_forman_ricci: bool = True
    perturbation_sigma: float = 0.1
    f2_max_amplification: float = 5.0
    knn_neighbors: int = 10
    f3_margin: int = 10

    # Validation Metrics
    compute_gromov_delta: bool = True
    compute_distortion: bool = True

    # Random Seed
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def default_config() -> ExperimentConfig:
    """
    Get default experiment configuration.

    Returns:
        ExperimentConfig with canonical hyperparameters.

    Notes:
        - This is the SINGLE SOURCE OF TRUTH
        - All experiments should start from this config
    """
    return ExperimentConfig()


def get_training_config(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """
    Convert ExperimentConfig to trainer-compatible dictionary.

    Args:
        config: Experiment configuration. Uses default if None.

    Returns:
        Dictionary for CGTTrainer initialization.
    """
    config = config or default_config()

    return {
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.num_epochs,
        "patience": config.patience,
        "weight_decay": config.weight_decay,
        "device": torch.device(config.device),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    EXPERIMENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def create_experiment(
    config: Optional[ExperimentConfig] = None,
) -> Tuple[Any, Any, Any, Dict]:
    """
    Create complete experiment setup.

    Args:
        config: Experiment configuration. Uses default if None.

    Returns:
        Tuple of (student, criterion, lipschitz_reg, training_config).

    Notes:
        - Imports here to avoid circular dependencies
        - Returns initialized but untrained model
    """
    from cgt.geometry import LorentzConfig
    from cgt.losses import MultiObjectiveLoss
    from cgt.models import CGTStudent
    from cgt.regularization import LipschitzRegularizer

    config = config or default_config()
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32

    # Get teacher dimension (assumed 384 for MiniLM)
    # In practice, this should be queried from the actual teacher model
    teacher_dim = 384  # all-MiniLM-L6-v2

    # Create student model
    student = CGTStudent(
        teacher_dim=teacher_dim,
        student_dim=config.hyperbolic_dim,
        hidden_dim=config.hidden_dim,
        learnable_curvature=config.learnable_curvature,
        initial_curvature=config.initial_curvature,
        curvature_min=config.curvature_min,
        curvature_max=config.curvature_max,
    ).to(device=device, dtype=dtype)

    # Initialize homeostatic field if enabled
    if config.enable_homeostatic:
        student.init_homeostatic(
            n_anchors=config.n_anchors,
            alpha=config.homeostatic_alpha,
        )

    # Create criterion
    criterion = MultiObjectiveLoss(
        lambda_contrastive=config.lambda_contrastive,
        lambda_distill=config.lambda_distill,
        lambda_spectral=config.lambda_spectral,
        lambda_topo=config.lambda_topo,
        lambda_lipschitz=config.lambda_lipschitz,
        radius_weight=config.radius_weight,
        temperature=config.temperature,
        target_beta_0=config.target_beta_0,
        power_law_alpha=config.power_law_alpha,
    ).to(device=device, dtype=dtype)

    # Create Lipschitz regularizer
    lipschitz_reg = LipschitzRegularizer(
        noise_scale=config.lipschitz_noise_scale,
    )

    # Training configuration
    training_config = get_training_config(config)

    return student, criterion, lipschitz_reg, training_config
