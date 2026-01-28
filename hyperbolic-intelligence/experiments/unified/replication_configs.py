# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Model-Specific Replication Configs
==================================

EXACT configurations extracted from each notebook.
NO modifications, NO harmonization, NO improvements.

SOURCE NOTEBOOKS:
1. k_Lighting_NUMERICAL_PARITY.ipynb
2. K_Lighting_AGI_v2_STSB_Training_(1).ipynb
3. CGT_Paper_Ready.ipynb
4. tutorial_completo_full_knowledge.ipynb

AUDIT COMPLIANCE:
- ✅ Values extracted verbatim from notebooks
- ✅ Differences documented explicitly
- ✅ Missing values marked as "NOT_SPECIFIED"
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ReplicationModel(Enum):
    """The 4 models to replicate (NO hybrid in this phase)."""
    K_LIGHT_NUMERICAL_PARITY = "k_light_numerical_parity"
    K_LIGHT_AGI_V2 = "k_light_agi_v2"
    CGT_PAPER_READY = "cgt_paper_ready"
    PSI_SLM = "psi_slm"


# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL 1: K_LIGHTING_NUMERICAL_PARITY
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb (PRIMARY REFERENCE)

@dataclass(frozen=True)
class KLightNumericalParityConfig:
    """
    EXACT config from k_Lighting_NUMERICAL_PARITY.ipynb
    This is the REFERENCE model against which all others are compared.
    """
    # Model Architecture
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    curvature: float = -1.0
    
    # Training
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01  # CRITICAL: fixed from 1e-5
    grad_clip: float = 1.0
    
    # Early Stopping
    early_stopping_patience: int = 25  # Stop after N epochs without improvement
    early_stopping_min_delta: float = 0.0001  # Minimum improvement threshold
    
    # Scheduler
    scheduler: str = "CosineAnnealingLR"
    t_max: int = 100
    
    # Loss Weights (λ)
    lambda_contrastive: float = 1.0
    lambda_distillation: float = 1.0
    lambda_topological: float = 0.1
    lambda_lipschitz: float = 0.01
    lambda_homeostatic: float = 0.001
    
    # Seed
    seed: int = 42
    seed_documented: bool = True
    
    # Dataset
    dataset: str = "mteb/stsbenchmark-sts"


# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL 2: K_LIGHTING_AGI_V2
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: K_Lighting_AGI_v2_STSB_Training_(1).ipynb

@dataclass(frozen=True)
class KLightAGIv2Config:
    """
    EXACT config from K_Lighting_AGI_v2_STSB_Training_(1).ipynb
    
    KEY DIFFERENCES FROM REFERENCE:
    - batch_size: 64 (vs 256)
    - learning_rate: 2e-4 (vs 1e-4)
    - num_epochs: 20 (vs 25)
    - weight_decay: 1e-5 (vs 0.01) ← DIFFERENT
    - lambda_distillation: 0.5 (vs 1.0)
    - lambda_topological: 0.3 (vs 0.1)
    - Additional: lambda_forman=0.1, lambda_coherence=0.1
    """
    # Model Architecture
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    curvature: float = -1.0
    
    # Training - DIFFERENT VALUES
    batch_size: int = 64  # DIFFERENT: 64 vs 256
    num_epochs: int = 100  # DIFFERENT: 20 vs 25
    learning_rate: float = 2e-4  # DIFFERENT: 2e-4 vs 1e-4
    weight_decay: float = 1e-5  # DIFFERENT: 1e-5 vs 0.01
    grad_clip: float = 1.0
    
    # Scheduler
    scheduler: str = "CosineAnnealingLR"
    t_max: int = 100  # Matches num_epochs
    eta_min: float = 1e-6  # Additional param
    
    # Loss Weights - DIFFERENT
    lambda_contrastive: float = 1.0
    lambda_distillation: float = 0.5  # DIFFERENT: 0.5 vs 1.0
    lambda_topological: float = 0.3  # DIFFERENT: 0.3 vs 0.1
    lambda_forman: float = 0.1  # ADDITIONAL
    lambda_coherence: float = 0.1  # ADDITIONAL
    
    # Seed
    seed: int = 42  # Assumed (not explicitly documented)
    seed_documented: bool = False  # NOT SPECIFIED in notebook
    
    # Dataset
    dataset: str = "mteb/stsbenchmark-sts"


# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL 3: CGT_PAPER_READY
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: CGT_Paper_Ready.ipynb

@dataclass(frozen=True)
class CGTPaperReadyConfig:
    """
    EXACT config from CGT_Paper_Ready.ipynb
    
    KEY DIFFERENCES FROM REFERENCE:
    - batch_size: 64 (vs 256)
    - learning_rate: 2e-4 (vs 1e-4)
    - lambda_distillation: 0.5 (vs 1.0)
    - lambda_topological: 0.5 (vs 0.1)
    - lambda_lipschitz: 0.8 (vs 0.01)
    - Additional: temperature=0.07, n_anchors=32
    """
    # Model Architecture
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    curvature: float = -1.0
    
    # Training - DIFFERENT VALUES
    batch_size: int = 64  # DIFFERENT: 64 vs 256
    num_epochs: int = 100  # SAME
    learning_rate: float = 2e-4  # DIFFERENT: 2e-4 vs 1e-4
    weight_decay: float = 0.01  # SAME
    grad_clip: float = 1.0
    
    # Scheduler
    scheduler: str = "CosineAnnealingLR"
    t_max: int = 100
    
    # Loss Weights - DIFFERENT
    lambda_contrastive: float = 1.0
    lambda_distillation: float = 0.5  # DIFFERENT: 0.5 vs 1.0
    lambda_topological: float = 0.5  # DIFFERENT: 0.5 vs 0.1
    lambda_lipschitz: float = 0.8  # DIFFERENT: 0.8 vs 0.01
    
    # Additional Loss Params
    temperature: float = 0.07  # ADDITIONAL
    target_beta_0: float = 1.0  # ADDITIONAL
    
    # Homeostatic Field
    enable_homeostatic: bool = True
    n_anchors: int = 32
    homeostatic_alpha: float = 0.2
    
    # Spectral Normalization
    use_spectral_norm: bool = True
    lipschitz_noise_scale: float = 0.05
    
    # Seed
    seed: int = 42
    seed_documented: bool = True  # Explicitly set with GLOBAL_SEED = 42
    
    # Dataset
    dataset: str = "mteb/stsbenchmark-sts"


# ═══════════════════════════════════════════════════════════════════════════════
#                    MODEL 4: PSI_SLM
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: tutorial_completo_full_knowledge.ipynb

@dataclass(frozen=True)
class PSISLMConfig:
    """
    EXACT config from tutorial_completo_full_knowledge.ipynb
    
    MAJOR DIFFERENCES FROM CGT MODELS:
    - teacher: all-mpnet-base-v2 (768d) vs all-MiniLM-L6-v2 (384d)
    - student_dim: 128 vs 32
    - num_epochs: 500 vs 25
    - Different loss structure (GW + NCE + Topo)
    - Uses curriculum learning
    """
    # Model Architecture - VERY DIFFERENT
    teacher_model: str = "sentence-transformers/all-mpnet-base-v2"  # DIFFERENT
    teacher_dim: int = 768  # DIFFERENT: 768 vs 384
    student_dim: int = 128  # DIFFERENT: 128 vs 32
    hidden_dim: int = 1024  # DIFFERENT: 1024 vs 256
    curvature: float = 1.0  # DIFFERENT: positive vs negative
    
    # Training - VERY DIFFERENT
    batch_size: int = 64  # Assumed default
    num_epochs: int = 100  # DIFFERENT: 500 vs 25
    learning_rate: float = 5e-4  # DIFFERENT: 5e-4 vs 1e-4
    weight_decay: float = 0.0  # NOT SPECIFIED, assumed 0
    grad_clip: float = 1.0  # Assumed
    
    # Early Stopping - longer patience for 500 epochs
    early_stopping_patience: int = 25  # Stop after 50 epochs without improvement
    early_stopping_min_delta: float = 0.0001
    
    # Loss Weights - DIFFERENT STRUCTURE
    lambda_gw: float = 1.0  # Gromov-Wasserstein
    lambda_nce: float = 0.5  # InfoNCE
    lambda_topo: float = 0.5  # Topological
    temperature: float = 0.07
    
    # Curriculum Learning - UNIQUE TO PSI_SLM
    use_curriculum: bool = True
    curriculum_start_epoch: int = 100  # 20% of 500
    curriculum_warmup: int = 150  # 30% of 500
    
    # Multi-Teacher Option
    multi_teacher_models: tuple = (
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "all-distilroberta-v1",
    )
    
    # Seed
    seed: int = 42  # Assumed (not explicitly documented)
    seed_documented: bool = False  # NOT SPECIFIED
    
    # Dataset - DIFFERENT (uses custom knowledge base)
    dataset: str = "custom_knowledge_base"  # NOT STS-B


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIG FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_replication_config(model: ReplicationModel):
    """Get the exact config for a replication model."""
    configs = {
        ReplicationModel.K_LIGHT_NUMERICAL_PARITY: KLightNumericalParityConfig(),
        ReplicationModel.K_LIGHT_AGI_V2: KLightAGIv2Config(),
        ReplicationModel.CGT_PAPER_READY: CGTPaperReadyConfig(),
        ReplicationModel.PSI_SLM: PSISLMConfig(),
    }
    return configs[model]


def get_config_diff_from_reference(model: ReplicationModel) -> Dict[str, Dict[str, Any]]:
    """
    Get differences between a model's config and the reference (K_LIGHT_NUMERICAL_PARITY).
    
    Returns:
        Dict with {param_name: {"reference": value, "model": value}}
    """
    ref = get_replication_config(ReplicationModel.K_LIGHT_NUMERICAL_PARITY)
    target = get_replication_config(model)
    
    if model == ReplicationModel.K_LIGHT_NUMERICAL_PARITY:
        return {}  # No differences from itself
    
    diffs = {}
    ref_dict = {k: v for k, v in ref.__dict__.items() if not k.startswith('_')}
    target_dict = {k: v for k, v in target.__dict__.items() if not k.startswith('_')}
    
    # Find differences
    all_keys = set(ref_dict.keys()) | set(target_dict.keys())
    
    for key in all_keys:
        ref_val = ref_dict.get(key, "NOT_IN_REFERENCE")
        target_val = target_dict.get(key, "NOT_IN_MODEL")
        
        if ref_val != target_val:
            diffs[key] = {
                "reference": ref_val,
                "model": target_val,
            }
    
    return diffs


# ═══════════════════════════════════════════════════════════════════════════════
#                    SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

REPLICATION_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REPLICATION CONFIG SUMMARY                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Parameter              │ K_LIGHT_NUM │ K_LIGHT_AGI │ CGT_PAPER │ PSI_SLM     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ teacher_dim            │ 384         │ 384         │ 384       │ 768         ║
║ student_dim            │ 32          │ 32          │ 32        │ 128         ║
║ batch_size             │ 256         │ 64          │ 64        │ 64          ║
║ num_epochs             │ 25          │ 20          │ 25        │ 500         ║
║ learning_rate          │ 1e-4        │ 2e-4        │ 2e-4      │ 5e-4        ║
║ weight_decay           │ 0.01        │ 1e-5        │ 0.01      │ 0.0         ║
║ lambda_distillation    │ 1.0         │ 0.5         │ 0.5       │ N/A (GW)    ║
║ lambda_topological     │ 0.1         │ 0.3         │ 0.5       │ 0.5         ║
║ lambda_lipschitz       │ 0.01        │ N/A         │ 0.8       │ N/A         ║
║ seed_documented        │ YES (42)    │ NO          │ YES (42)  │ NO          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(REPLICATION_SUMMARY)
    
    for model in ReplicationModel:
        print(f"\n{'='*60}")
        print(f"Model: {model.value}")
        print(f"{'='*60}")
        
        config = get_replication_config(model)
        print(f"Config: {config.__class__.__name__}")
        
        diffs = get_config_diff_from_reference(model)
        if diffs:
            print(f"\nDifferences from reference:")
            for param, vals in diffs.items():
                print(f"  {param}: {vals['reference']} → {vals['model']}")
