# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Hybrid Model Configuration
==========================

EXPLICIT combination of existing components ONLY.
NO new losses. NO new architectures. NO invented weights.

HYBRID DEFINITION:
- Architecture base: K-Lighting Numerical Parity
- Teacher: Ψ-SLM (all-mpnet-base-v2, 768d)
- Losses: Contrastive + Distillation + Topological + Forman-Ricci + Lipschitz
- NOT included: Homeostatic

AUDIT COMPLIANCE:
- ✅ Each component has explicit origin comment
- ✅ No new losses created
- ✅ No hyperparameter tuning
- ✅ Weights from documented sources only
"""

from dataclasses import dataclass
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
#                    HYBRID MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HybridModelConfig:
    """
    Hybrid model combining components from existing models.
    
    ARCHITECTURE:
    - Base: K-Lighting Numerical Parity (k_Lighting_NUMERICAL_PARITY.ipynb)
    - Student: CGTStudentHardened (src/cgt/models/cgt_hardened.py)
    - Substrate: LorentzSubstrateHardened (src/cgt/geometry/lorentz_hardened.py)
    
    TEACHER:
    - Model: all-mpnet-base-v2 (tutorial_completo_full_knowledge.ipynb - PSI_SLM)
    - Dimension: 768 (vs 384 in K-Lighting)
    
    LOSSES (with origins):
    - Contrastive: K-Lighting (k_Lighting_NUMERICAL_PARITY.ipynb)
    - Distillation: K-Lighting (k_Lighting_NUMERICAL_PARITY.ipynb)
    - Topological β₀: K-Lighting (k_Lighting_NUMERICAL_PARITY.ipynb)
    - Forman-Ricci: K-Lighting AGI v2 (K_Lighting_AGI_v2_STSB_Training.ipynb)
    - Lipschitz: CGT Paper Ready (CGT_Paper_Ready.ipynb)
    
    NOT INCLUDED:
    - Homeostatic (explicitly excluded per specification)
    - GW loss (PSI_SLM specific, not compatible)
    - Coherence loss (AGI v2 specific)
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    ARCHITECTURE (from K-Lighting Numerical Parity)
    # ═══════════════════════════════════════════════════════════════════════════
    # SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    
    student_dim: int = 32  # From K-Lighting Numerical Parity
    hidden_dim: int = 256  # From K-Lighting Numerical Parity
    curvature: float = -1.0  # Lorentz manifold (K-Lighting)
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    TEACHER (from PSI_SLM)
    # ═══════════════════════════════════════════════════════════════════════════
    # SOURCE: tutorial_completo_full_knowledge.ipynb
    
    teacher_model: str = "sentence-transformers/all-mpnet-base-v2"  # From PSI_SLM
    teacher_dim: int = 768  # all-mpnet-base-v2 dimension
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    TRAINING HYPERPARAMETERS (from K-Lighting Numerical Parity)
    # ═══════════════════════════════════════════════════════════════════════════
    # SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    
    batch_size: int = 256  # From K-Lighting Numerical Parity
    num_epochs: int = 100  # From K-Lighting Numerical Parity
    learning_rate: float = 1e-4  # From K-Lighting Numerical Parity
    weight_decay: float = 0.01  # From K-Lighting Numerical Parity (CRITICAL)
    grad_clip: float = 1.0  # From K-Lighting Numerical Parity
    
    # Early Stopping
    early_stopping_patience: int = 25  # Stop after N epochs without improvement
    early_stopping_min_delta: float = 0.0001  # Minimum improvement threshold
    
    # Scheduler
    scheduler: str = "CosineAnnealingLR"  # From K-Lighting Numerical Parity
    t_max: int = 100  # From K-Lighting Numerical Parity
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    LOSS WEIGHTS (COMBINED FROM MULTIPLE SOURCES)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Contrastive Loss
    # SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    lambda_contrastive: float = 1.0
    
    # Distillation Loss
    # SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    lambda_distillation: float = 1.0
    
    # Topological Loss (β₀)
    # SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    lambda_topological: float = 0.1
    
    # Forman-Ricci Loss
    # SOURCE: K_Lighting_AGI_v2_STSB_Training_(1).ipynb
    lambda_forman: float = 0.1
    
    # Lipschitz Regularization
    # SOURCE: CGT_Paper_Ready.ipynb
    lambda_lipschitz: float = 0.8  # CGT Paper Ready uses 0.8
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    EXPLICITLY EXCLUDED
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Homeostatic Loss - EXCLUDED per specification
    # SOURCE: Would be from CGT_Paper_Ready.ipynb but NOT included
    enable_homeostatic: bool = False
    lambda_homeostatic: float = 0.0  # Explicitly zero
    
    # Coherence Loss - NOT included (AGI v2 specific)
    lambda_coherence: float = 0.0  # Explicitly zero
    
    # GW Loss - NOT included (PSI_SLM specific, incompatible)
    lambda_gw: float = 0.0  # Explicitly zero
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                    SEED AND DATASET
    # ═══════════════════════════════════════════════════════════════════════════
    
    seed: int = 42  # Fixed seed as specified
    dataset: str = "mteb/stsbenchmark-sts"  # Standard STS-B


# ═══════════════════════════════════════════════════════════════════════════════
#                    COMPONENT ORIGIN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

HYBRID_COMPONENT_ORIGINS: Dict[str, Dict[str, Any]] = {
    "architecture": {
        "component": "CGTStudentHardened + LorentzSubstrateHardened",
        "source_notebook": "k_Lighting_NUMERICAL_PARITY.ipynb",
        "source_module": "src/cgt/models/cgt_hardened.py",
        "parameters": {
            "student_dim": 32,
            "hidden_dim": 256,
            "curvature": -1.0,
        },
    },
    
    "teacher": {
        "component": "all-mpnet-base-v2",
        "source_notebook": "tutorial_completo_full_knowledge.ipynb (PSI_SLM)",
        "source_module": "sentence-transformers",
        "parameters": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
        },
        "justification": "Larger teacher (768d) provides richer supervision than MiniLM (384d)",
    },
    
    "loss_contrastive": {
        "component": "InfoNCE-style contrastive loss",
        "source_notebook": "k_Lighting_NUMERICAL_PARITY.ipynb",
        "source_module": "src/cgt/losses/losses_hardened.py",
        "weight": 1.0,
    },
    
    "loss_distillation": {
        "component": "Teacher-student alignment loss",
        "source_notebook": "k_Lighting_NUMERICAL_PARITY.ipynb",
        "source_module": "src/cgt/losses/losses_hardened.py",
        "weight": 1.0,
    },
    
    "loss_topological": {
        "component": "Betti number (β₀) preservation",
        "source_notebook": "k_Lighting_NUMERICAL_PARITY.ipynb",
        "source_module": "src/cgt/losses/losses_hardened.py",
        "weight": 0.1,
    },
    
    "loss_forman": {
        "component": "Forman-Ricci curvature regularization",
        "source_notebook": "K_Lighting_AGI_v2_STSB_Training_(1).ipynb",
        "source_module": "src/cgt/losses/losses_hardened.py",
        "weight": 0.1,
    },
    
    "loss_lipschitz": {
        "component": "Lipschitz gradient penalty",
        "source_notebook": "CGT_Paper_Ready.ipynb",
        "source_module": "src/cgt/losses/losses_hardened.py",
        "weight": 0.8,
    },
    
    "training_hyperparameters": {
        "component": "Training configuration",
        "source_notebook": "k_Lighting_NUMERICAL_PARITY.ipynb",
        "parameters": {
            "batch_size": 256,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                    EXCLUDED COMPONENTS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

HYBRID_EXCLUDED_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "homeostatic": {
        "component": "Homeostatic field regularization",
        "source_notebook": "CGT_Paper_Ready.ipynb",
        "reason": "Explicitly excluded per specification",
    },
    
    "coherence": {
        "component": "Coherence loss",
        "source_notebook": "K_Lighting_AGI_v2_STSB_Training_(1).ipynb",
        "reason": "Model-specific, not part of hybrid definition",
    },
    
    "gw_loss": {
        "component": "Gromov-Wasserstein loss",
        "source_notebook": "tutorial_completo_full_knowledge.ipynb",
        "reason": "PSI_SLM specific, incompatible with standard training loop",
    },
    
    "curriculum_learning": {
        "component": "Curriculum learning schedule",
        "source_notebook": "tutorial_completo_full_knowledge.ipynb",
        "reason": "PSI_SLM specific, not part of K-Lighting base",
    },
}


def get_hybrid_config() -> HybridModelConfig:
    """Get the hybrid model configuration."""
    return HybridModelConfig()


def get_hybrid_summary() -> str:
    """Get a human-readable summary of the hybrid model."""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HYBRID MODEL DEFINITION                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ARCHITECTURE BASE: K-Lighting Numerical Parity                              ║
║  ├── Student: CGTStudentHardened (32d output)                                ║
║  ├── Substrate: LorentzSubstrateHardened (c=-1.0)                            ║
║  └── Hidden: 256d MLP                                                        ║
║                                                                              ║
║  TEACHER: PSI_SLM (all-mpnet-base-v2)                                        ║
║  └── Dimension: 768 (vs 384 in original K-Lighting)                          ║
║                                                                              ║
║  LOSSES (with weights and origins):                                          ║
║  ├── Contrastive     λ=1.0   [K-Lighting Numerical Parity]                   ║
║  ├── Distillation    λ=1.0   [K-Lighting Numerical Parity]                   ║
║  ├── Topological β₀  λ=0.1   [K-Lighting Numerical Parity]                   ║
║  ├── Forman-Ricci    λ=0.1   [K-Lighting AGI v2]                             ║
║  └── Lipschitz       λ=0.8   [CGT Paper Ready]                               ║
║                                                                              ║
║  EXCLUDED:                                                                   ║
║  ├── Homeostatic (per specification)                                         ║
║  ├── Coherence (AGI v2 specific)                                             ║
║  └── GW loss (PSI_SLM specific)                                              ║
║                                                                              ║
║  TRAINING:                                                                   ║
║  ├── Batch: 256, Epochs: 25, LR: 1e-4                                        ║
║  ├── Weight decay: 0.01                                                      ║
║  ├── Scheduler: CosineAnnealingLR                                            ║
║  └── Seed: 42                                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(get_hybrid_summary())
    
    print("\nComponent Origins:")
    print("=" * 60)
    for name, info in HYBRID_COMPONENT_ORIGINS.items():
        print(f"\n{name}:")
        print(f"  Source: {info['source_notebook']}")
        if 'weight' in info:
            print(f"  Weight: {info['weight']}")
    
    print("\n\nExcluded Components:")
    print("=" * 60)
    for name, info in HYBRID_EXCLUDED_COMPONENTS.items():
        print(f"\n{name}:")
        print(f"  Reason: {info['reason']}")
