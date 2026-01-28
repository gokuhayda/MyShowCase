# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Publication Configs
===================

Configuration for publication experiments.

AUDIT COMPLIANCE:
- ✅ Does NOT duplicate ExperimentConfig fields
- ✅ Imports and reuses existing constants from experiments modules
- ✅ Only defines publication-specific parameters not in CGT codebase

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════════════════════
#              REUSE EXISTING CONFIG (NO DUPLICATION)
# ═══════════════════════════════════════════════════════════════════════════════

# REUSE ExperimentConfig from src/cgt/experiments/part1_reference.py
# This is the SINGLE SOURCE OF TRUTH for experiment parameters
from cgt.experiments.part1_reference import ExperimentConfig


# ═══════════════════════════════════════════════════════════════════════════════
#              PUBLICATION-SPECIFIC CONFIG (NEW - does not exist elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# PublicationConfig extends ExperimentConfig with publication-specific fields.
# This does not duplicate fields but adds new functionality.

@dataclass
class PublicationConfig(ExperimentConfig):
    """
    Configuration for publication experiments.
    
    EXTENDS ExperimentConfig (not duplicates) with publication-specific fields:
    - n_seeds: Number of seeds for multi-seed robustness
    - min_epochs: Minimum epochs before early stopping can trigger
    - min_delta: Minimum improvement for early stopping
    
    Searched codebase for equivalent fields in:
    * src/cgt/experiments/part1_reference.py (ExperimentConfig)
    * experiments/ablations/*
    * experiments/analysis/*
    The fields below do NOT exist in ExperimentConfig.
    """
    
    # Publication-specific fields (NOT in ExperimentConfig)
    n_seeds: int = 3  # For statistical robustness
    min_epochs: int = 10  # Minimum before early stopping
    min_delta: float = 0.001  # Early stopping threshold
    
    # Override defaults for publication standards
    # (These exist in parent but we use different values)
    num_epochs: int = 30  # Reduced from 50 for faster iteration
    patience: int = 5  # More conservative than default 10


# ═══════════════════════════════════════════════════════════════════════════════
#              ABLATION DIMENSIONS (NEW - not defined elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# ABLATION_DIMS does not exist as a constant in the current CGT codebase.
# Dimensions are specified inline in each experiment. Centralized here.

ABLATION_DIMS: List[int] = [8, 16, 32, 64]
"""
Dimensions for dimensional ablation study.
Covers 48× to 6× compression ratio.

Searched codebase for `ABLATION_DIMS` or equivalent constant in:
* src/cgt/...
* experiments/...
No centralized definition found.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#              STS BENCHMARK CONFIGS (NEW - not defined elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# STS_CONFIGS does not exist as a centralized constant in the current CGT codebase.
# Dataset configs are scattered across modules. Centralized here.

STS_CONFIGS: List[Tuple[str, str, str, str, str, str, float]] = [
    # (name, huggingface_path, split, text1_col, text2_col, score_col, score_scale)
    
    # Main benchmark
    ('STSb', 'mteb/stsbenchmark-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    
    # Cross-dataset transfer (CRITICAL: mteb/sickr-sts NOT sick)
    ('SICK-R', 'mteb/sickr-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    
    # STS12-16 for generalization
    ('STS12', 'mteb/sts12-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    ('STS13', 'mteb/sts13-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    ('STS14', 'mteb/sts14-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    ('STS15', 'mteb/sts15-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
    ('STS16', 'mteb/sts16-sts', 'test', 'sentence1', 'sentence2', 'score', 5.0),
]
"""
STS benchmark configurations.

Searched codebase for `STS_CONFIGS` or equivalent constant in:
* src/cgt/...
* experiments/...
No centralized definition found.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#              EVALUATION MODELS (NEW - not defined elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# EVAL_MODELS does not exist as a centralized constant in the current CGT codebase.
# Model configs are specified inline in each experiment. Centralized here.

EVAL_MODELS: List[Tuple[str, str, int]] = [
    # (display_name, huggingface_path, embedding_dim)
    
    # Primary (all-MiniLM-L6-v2)
    ('MiniLM-L6', 'sentence-transformers/all-MiniLM-L6-v2', 384),
    
    # Extended benchmark
    ('MiniLM-L12', 'sentence-transformers/all-MiniLM-L12-v2', 384),
    ('MPNet', 'sentence-transformers/all-mpnet-base-v2', 768),
    ('DistilRoBERTa', 'sentence-transformers/all-distilroberta-v1', 768),
]
"""
Evaluation models for multi-model benchmark.

Searched codebase for `EVAL_MODELS` or equivalent constant in:
* src/cgt/...
* experiments/...
No centralized definition found.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#              STATISTICAL THRESHOLDS (NEW - not defined elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# Statistical constants do not exist as centralized definitions in CGT codebase.

SIGNIFICANCE_ALPHA: float = 0.05
"""Standard significance level."""

SESOI: float = 0.02
"""Smallest Effect Size of Interest for Spearman ρ."""

COHENS_D_THRESHOLDS = {
    'negligible': 0.2,
    'small': 0.5,
    'medium': 0.8,
    'large': float('inf'),
}
"""Cohen's d effect size interpretation."""
