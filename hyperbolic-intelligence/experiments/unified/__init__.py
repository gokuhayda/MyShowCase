# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Unified Experiment Module
=========================

New unified runner for all 5 models.
Does NOT modify existing publication runner.

Components:
- config.py: Unified configuration for all models
- trainer.py: Unified training interface
- unified_runner.py: Main experiment runner
- replication_configs.py: Exact configs for 4 original models
- replication_executor.py: Faithful replication executor
"""

from .config import (
    ModelType,
    RunMode,
    ExecutionConfig,
    TrainingHyperparameters,
    LossWeights,
    ModelArchitecture,
    ModelConfig,
    get_model_configs,
    validate_config,
)

from .trainer import UnifiedTrainer

from .unified_runner import (
    UnifiedRunner,
    run_experiment,
    enforce_float64,
    get_device_info,
    load_stsb_data,
    train_single_model,
)

from .replication_configs import (
    ReplicationModel,
    get_replication_config,
    get_config_diff_from_reference,
    KLightNumericalParityConfig,
    KLightAGIv2Config,
    CGTPaperReadyConfig,
    PSISLMConfig,
)

from .replication_executor import (
    ReplicationTrainer,
    run_all_replications,
)

from .hybrid_config import (
    HybridModelConfig,
    get_hybrid_config,
    get_hybrid_summary,
    HYBRID_COMPONENT_ORIGINS,
    HYBRID_EXCLUDED_COMPONENTS,
)

from .hybrid_executor import (
    HybridTrainer,
    train_hybrid,
    load_hybrid_data,
)

from .evaluation import (
    UnifiedEvaluator,
    EvaluationResult,
    compute_spearman,
    compute_pearson,
    f1_projection_integrity,
    f2_distance_preservation,
    f3_topological_consistency,
)

from .final_executor import (
    FinalExecutor,
    run_final_execution,
    load_data_for_evaluation,
    load_trained_model,
)

# Hybrid Active Loss (new)
from .hybrid_active_trainer import (
    HybridActiveTrainer,
    train_hybrid_active,
)

# === PATCH: PSI-SLM Full Trainer ===
try:
    from .psi_slm_trainer import PsiSlmFullTrainer, PsiSlmLoss
    HAS_PSI_SLM_TRAINER = True
except ImportError:
    HAS_PSI_SLM_TRAINER = False

__all__ = [
    # Config
    "ModelType",
    "RunMode",
    "ExecutionConfig",
    "TrainingHyperparameters",
    "LossWeights",
    "ModelArchitecture",
    "ModelConfig",
    "get_model_configs",
    "validate_config",
    # Trainer
    "UnifiedTrainer",
    # Runner
    "UnifiedRunner",
    "run_experiment",
    "enforce_float64",
    "get_device_info",
    "load_stsb_data",
    "train_single_model",
    # Replication
    "ReplicationModel",
    "get_replication_config",
    "get_config_diff_from_reference",
    "KLightNumericalParityConfig",
    "KLightAGIv2Config",
    "CGTPaperReadyConfig",
    "PSISLMConfig",
    "ReplicationTrainer",
    "run_all_replications",
    # Hybrid
    "HybridModelConfig",
    "get_hybrid_config",
    "get_hybrid_summary",
    "HYBRID_COMPONENT_ORIGINS",
    "HYBRID_EXCLUDED_COMPONENTS",
    "HybridTrainer",
    "train_hybrid",
    "load_hybrid_data",
    # Evaluation
    "UnifiedEvaluator",
    "EvaluationResult",
    "compute_spearman",
    "compute_pearson",
    "f1_projection_integrity",
    "f2_distance_preservation",
    "f3_topological_consistency",
    # Final Execution
    "FinalExecutor",
    "run_final_execution",
    "load_data_for_evaluation",
    "load_trained_model",
    # Hybrid Active (new)
    "HybridActiveTrainer",
    "train_hybrid_active",
    # === PATCH: PSI-SLM Full ===
    "PsiSlmFullTrainer",
    "PsiSlmLoss",
]
