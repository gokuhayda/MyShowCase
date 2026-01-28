# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Publication Module
======================

Minimal orchestration layer for publication-ready experiments.
All logic delegated to existing modules - this is ENCAPSULATION ONLY.

AUDIT COMPLIANCE:
- ✅ No duplicated functions (uses imports from cgt.utils)
- ✅ No duplicated configs (extends ExperimentConfig)
- ✅ Re-exports existing functions instead of reimplementing

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from .runner import (
    # NEW functions (verified non-existent elsewhere)
    log_environment,
    create_results_directory,
    setup_experiment,
    # Orchestration class
    PublicationRunner,
)

from .configs import (
    # EXTENDS ExperimentConfig (not duplicates)
    PublicationConfig,
    # NEW constants (verified non-existent elsewhere)
    ABLATION_DIMS,
    STS_CONFIGS,
    EVAL_MODELS,
    SIGNIFICANCE_ALPHA,
    SESOI,
    COHENS_D_THRESHOLDS,
)

from .outputs import (
    # NEW functions (verified non-existent elsewhere)
    save_metrics,
    to_latex_table,
    save_latex_table,
    setup_publication_style,
    save_figure,
    plot_comparison_bar,
    generate_summary,
    print_summary_table,
    COLORS,
)

# RE-EXPORT from existing modules (NOT reimplemented)
# These are the canonical implementations
from cgt.utils.helpers import set_global_seed, get_device, clear_memory
from experiments.run_all_experiments import compute_teacher_spearman

__all__ = [
    # Runner (NEW)
    "log_environment",
    "create_results_directory", 
    "setup_experiment",
    "PublicationRunner",
    # Configs (EXTENDS existing)
    "PublicationConfig",
    "ABLATION_DIMS",
    "STS_CONFIGS",
    "EVAL_MODELS",
    "SIGNIFICANCE_ALPHA",
    "SESOI",
    "COHENS_D_THRESHOLDS",
    # Outputs (NEW)
    "save_metrics",
    "to_latex_table",
    "save_latex_table",
    "setup_publication_style",
    "save_figure",
    "plot_comparison_bar",
    "generate_summary",
    "print_summary_table",
    "COLORS",
    # RE-EXPORTED from existing (CANONICAL source)
    "set_global_seed",
    "get_device",
    "clear_memory",
    "compute_teacher_spearman",
]
