# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unified Experiment Configuration
================================

Contains EXACT specifications for all 5 models extracted from the original notebooks.
This file serves as the single source of truth for hyperparameters.

SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb (primary reference)

AUDIT COMPLIANCE:
- ✅ Hyperparameters extracted from audited notebooks
- ✅ No modification of original values
- ✅ float64 enforced end-to-end
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
#                              MODEL TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ModelType(Enum):
    """Enumeration of the 6 models to be compared."""
    K_LIGHT_NUMERICAL_PARITY = "k_lighting_numerical_parity"
    K_LIGHT_AGI_V2 = "k_lighting_agi_v2"
    CGT_PAPER_READY = "cgt_paper_ready"
    PSI_SLM = "psi_slm"
    HYBRID = "hybrid"
    # === PATCH: train_toy integration via psi_extensions ===
    PSI_SLM_FULL = "psi_slm_full"  # Full Ψ-SLM with H-AKOrN + Topological Loss


class RunMode(Enum):
    """Execution modes for the runner."""
    SINGLE = "single"              # Run one model at a time
    ALL_SEQUENTIAL = "all_sequential"  # Run all models sequentially
    ALL_PARALLEL = "all_parallel"  # Run all models in parallel processes


# ═══════════════════════════════════════════════════════════════════════════════
#                         HYPERPARAMETERS (FROM AUDIT)
# ═══════════════════════════════════════════════════════════════════════════════

# Source: k_Lighting_NUMERICAL_PARITY.ipynb (Cell 10-15)
# These values are FROZEN and must not be modified

@dataclass(frozen=True)
class TrainingHyperparameters:
    """
    Training hyperparameters extracted from k_Lighting_NUMERICAL_PARITY.ipynb.
    
    FROZEN: These values cannot be changed after creation.
    """
    batch_size: int = 256
    num_epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 0.01  # CRITICAL: was 1e-5 in buggy versions
    grad_clip: float = 1.0
    seed: int = 42
    
    # Scheduler
    scheduler_type: str = "cosine"
    t_max: int = 25  # Same as num_epochs


@dataclass(frozen=True)
class LossWeights:
    """
    Loss weights (λ) extracted from k_Lighting_NUMERICAL_PARITY.ipynb.
    
    FROZEN: These values cannot be changed after creation.
    """
    contrastive: float = 1.0
    distillation: float = 1.0
    topological: float = 0.1
    lipschitz: float = 0.01
    homeostatic: float = 0.001


@dataclass(frozen=True)
class ModelArchitecture:
    """
    Model architecture parameters extracted from notebooks.
    
    FROZEN: These values cannot be changed after creation.
    """
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    curvature: float = -1.0  # Lorentz manifold


# ═══════════════════════════════════════════════════════════════════════════════
#                         MODEL-SPECIFIC CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_type: ModelType
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    architecture: ModelArchitecture = field(default_factory=ModelArchitecture)
    
    # Model-specific overrides
    description: str = ""
    enabled: bool = True
    skip_reason: Optional[str] = None


def get_model_configs() -> Dict[ModelType, ModelConfig]:
    """
    Get configurations for all 5 models.
    
    Returns:
        Dictionary mapping ModelType to ModelConfig
    """
    return {
        ModelType.K_LIGHT_NUMERICAL_PARITY: ModelConfig(
            model_type=ModelType.K_LIGHT_NUMERICAL_PARITY,
            description="Primary reference model from k_Lighting_NUMERICAL_PARITY.ipynb",
            enabled=True,
        ),
        
        ModelType.K_LIGHT_AGI_V2: ModelConfig(
            model_type=ModelType.K_LIGHT_AGI_V2,
            description="Alternative training from K_Lighting_AGI_v2_STSB_Training.ipynb",
            enabled=True,
        ),
        
        ModelType.CGT_PAPER_READY: ModelConfig(
            model_type=ModelType.CGT_PAPER_READY,
            description="Publication-ready model from CGT_Paper_Ready.ipynb",
            enabled=True,
        ),
        
        ModelType.PSI_SLM: ModelConfig(
            model_type=ModelType.PSI_SLM,
            description="ψ-SLM integration (optional)",
            enabled=False,  # Skip by default
            skip_reason="Requires ψ-SLM dependencies",
        ),
        
        ModelType.HYBRID: ModelConfig(
            model_type=ModelType.HYBRID,
            description="Hybrid combination of existing losses",
            enabled=True,
        ),
        
        # === PATCH: PSI_SLM_FULL integration ===
        ModelType.PSI_SLM_FULL: ModelConfig(
            model_type=ModelType.PSI_SLM_FULL,
            description="Full Ψ-SLM with H-AKOrN binding and topological loss (psi_extensions)",
            enabled=True,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                         EXECUTION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionConfig:
    """Configuration for experiment execution."""
    
    # Run mode
    mode: RunMode = RunMode.ALL_SEQUENTIAL
    
    # Specific model to run (only used in SINGLE mode)
    target_model: Optional[ModelType] = None
    
    # Output directories
    output_base: Path = field(default_factory=lambda: Path("./outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    
    # Precision (CRITICAL)
    dtype: torch.dtype = torch.float64
    
    # Device
    device: Optional[str] = None  # None = auto-detect
    
    # Parallelism
    max_workers: int = 4  # For parallel mode
    
    # Skip PSI_SLM by default
    skip_psi_slm: bool = True
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                         VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_config(config: ExecutionConfig) -> List[str]:
    """
    Validate execution configuration.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check dtype
    if config.dtype != torch.float64:
        errors.append(f"dtype must be torch.float64, got {config.dtype}")
    
    # Check mode consistency
    if config.mode == RunMode.SINGLE and config.target_model is None:
        errors.append("SINGLE mode requires target_model to be specified")
    
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#                         CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Expected performance from k_Lighting_NUMERICAL_PARITY.ipynb
EXPECTED_VAL_RHO = 0.793
EXPECTED_VAL_RHO_TOLERANCE = 0.005

# Dataset
STSB_DATASET = "mteb/stsbenchmark-sts"
STSB_SPLITS = ("train", "validation", "test")
