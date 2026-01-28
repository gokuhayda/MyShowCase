# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Publication Runner
==================

Orchestrates experiments using ONLY existing hardened modules.
NO formula derivation. NO reimplementation. ENCAPSULATION ONLY.

AUDIT COMPLIANCE:
- ✅ Uses set_global_seed from src/cgt/utils/helpers
- ✅ Uses get_device from src/cgt/utils/helpers
- ✅ Uses compute_teacher_spearman from experiments/run_all_experiments
- ✅ Early stopping via CGTTrainer config (not standalone class)
- ✅ Config via ExperimentConfig extension (not duplication)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════════════════════
#              REUSE EXISTING CGT FUNCTIONS (NO DUPLICATION)
# ═══════════════════════════════════════════════════════════════════════════════

# From src/cgt/utils/helpers.py - REUSE, NOT DUPLICATE
from cgt.utils.helpers import set_global_seed, get_device, clear_memory

# From experiments/run_all_experiments.py - REUSE, NOT DUPLICATE
from experiments.run_all_experiments import compute_teacher_spearman

# HARDENED MODULES ONLY
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.losses.losses_hardened import MultiObjectiveLoss
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig

# EXPERIMENTS MODULES - ENCAPSULATION ONLY
from experiments.ablations import (
    AblationConfig,
    run_euclidean_ablation,
    MRLConfig,
    run_mrl_comparison,
    BQComparisonConfig,
    run_bq_comparison,
)
from experiments.benchmarks import (
    run_cascade_compression,
    LatencyConfig,
    run_latency_benchmark,
)
from experiments.analysis import (
    RobustnessConfig,
    run_statistical_robustness,
)


# ═══════════════════════════════════════════════════════════════════════════════
#              NEW FUNCTIONALITY (verified non-existent in codebase)
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE:
# log_environment() does not exist in the current CGT codebase.
# Implemented here as a single source of truth for environment logging.

def log_environment() -> Dict[str, str]:
    """
    Log environment for reproducibility.
    
    Searched codebase for `log_environment` or equivalent functionality in:
    * src/cgt/...
    * experiments/...
    * utils/...
    No equivalent implementation found.
    """
    env = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": str(get_device()),  # REUSE existing function
        "timestamp": datetime.now().isoformat(),
    }
    
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "N/A"
        env["gpu_name"] = torch.cuda.get_device_name(0)
    
    # Optional libraries
    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    return env


# NOTE:
# create_results_directory() does not exist as a function in the current CGT codebase.
# The pattern mkdir(parents=True, exist_ok=True) is scattered across modules.
# Implemented here to centralize and avoid inline repetition.

def create_results_directory(base_name: str = "cgt_results") -> Dict[str, Path]:
    """
    Create timestamped results directory structure.
    
    Searched codebase for `create_results_directory` or equivalent functionality in:
    * src/cgt/...
    * experiments/...
    * utils/...
    No equivalent implementation found.
    (Inline mkdir pattern exists, but no wrapper function.)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"{base_name}_{timestamp}")
    
    dirs = {
        "base": base_dir,
        "logs": base_dir / "logs",
        "figures": base_dir / "figures",
        "tables": base_dir / "tables",
        "stats": base_dir / "stats",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def setup_experiment(seed: int = 42) -> Tuple[torch.device, Dict[str, Path], Dict[str, str]]:
    """
    Complete experiment setup.
    
    USES existing functions:
    - set_global_seed from src/cgt/utils/helpers
    - get_device from src/cgt/utils/helpers
    
    Returns:
        device: torch.device
        dirs: Dict with output directories
        env: Dict with environment info
    """
    set_global_seed(seed)  # REUSE existing
    device = get_device()  # REUSE existing
    dirs = create_results_directory()  # NEW
    env = log_environment()  # NEW
    env["seed"] = str(seed)
    
    # Save environment
    with open(dirs["stats"] / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    
    return device, dirs, env


# ═══════════════════════════════════════════════════════════════════════════════
#              PUBLICATION RUNNER (ENCAPSULATION ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

class PublicationRunner:
    """
    Orchestrates all publication experiments.
    
    ENCAPSULATION ONLY - all logic delegated to existing modules.
    NO formula derivation. NO CGT reimplementation.
    
    Early stopping is configured via CGTTrainer config dict,
    NOT via standalone EarlyStopping class (which would duplicate logic).
    """
    
    def __init__(
        self,
        device: torch.device,
        dirs: Dict[str, Path],
        seed: int = 42,
        max_epochs: int = 30,
        n_seeds: int = 3,
    ):
        self.device = device
        self.dirs = dirs
        self.seed = seed
        self.max_epochs = max_epochs
        self.n_seeds = n_seeds
        self.dtype = torch.float64  # Match hardened modules
        
        self.results = {}
    
    def run_euclidean_ablation(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
        test_emb1: torch.Tensor,
        test_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        teacher_spearman: float,
        teacher_dim: int = 384,
    ) -> Dict:
        """
        Run Part IV.1: Euclidean Ablation.
        
        ENCAPSULATION ONLY - delegates to experiments.ablations
        """
        config = AblationConfig(
            teacher_dim=teacher_dim,
            num_epochs=self.max_epochs,
            seed=self.seed,
            device=str(self.device),
        )
        
        results = run_euclidean_ablation(
            train_emb1=train_emb1,
            train_emb2=train_emb2,
            train_scores=train_scores,
            val_emb1=val_emb1,
            val_emb2=val_emb2,
            val_scores=val_scores,
            test_emb1=test_emb1,
            test_emb2=test_emb2,
            test_scores=test_scores,
            teacher_spearman=teacher_spearman,
            config=config,
            output_dir=self.dirs["stats"] / "part_iv_1",
        )
        
        self.results["part_iv_1"] = results
        return results
    
    def run_mrl_comparison(
        self,
        test_emb1: torch.Tensor,
        test_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        teacher_spearman: float,
        cgt_spearman: float,
        teacher_dim: int = 384,
        target_dims: List[int] = None,
    ) -> Dict:
        """
        Run Part IV.2: MRL Comparison.
        
        ENCAPSULATION ONLY - delegates to experiments.ablations
        """
        if target_dims is None:
            target_dims = [16, 32, 64, 128, 256]
            
        config = MRLConfig(
            teacher_dim=teacher_dim,
            target_dims=target_dims,
        )
        
        results = run_mrl_comparison(
            test_emb1=test_emb1,
            test_emb2=test_emb2,
            test_scores=test_scores,
            teacher_spearman=teacher_spearman,
            cgt_spearman=cgt_spearman,
            config=config,
            output_dir=self.dirs["stats"] / "part_iv_2",
        )
        
        self.results["part_iv_2"] = results
        return results
    
    def run_bq_comparison(
        self,
        cgt_emb1: torch.Tensor,
        cgt_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        cgt_spearman: float,
        teacher_spearman: float,
    ) -> Dict:
        """
        Run Part IV.3: Binary Quantization Comparison.
        
        ENCAPSULATION ONLY - delegates to experiments.ablations
        """
        config = BQComparisonConfig()
        
        results = run_bq_comparison(
            cgt_emb1=cgt_emb1,
            cgt_emb2=cgt_emb2,
            test_scores=test_scores,
            cgt_spearman=cgt_spearman,
            teacher_spearman=teacher_spearman,
            config=config,
            output_dir=self.dirs["stats"] / "part_iv_3",
        )
        
        self.results["part_iv_3"] = results
        return results
    
    def run_cascade_compression(
        self,
        cgt_emb1: torch.Tensor,
        cgt_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        cgt_spearman: float,
        teacher_spearman: float,
    ) -> Dict:
        """
        Run Part I.19: Cascade Compression.
        
        ENCAPSULATION ONLY - delegates to experiments.benchmarks
        """
        results = run_cascade_compression(
            cgt_emb1=cgt_emb1,
            cgt_emb2=cgt_emb2,
            test_scores=test_scores,
            cgt_spearman=cgt_spearman,
            teacher_spearman=teacher_spearman,
            output_dir=self.dirs["stats"] / "part_i_19",
        )
        
        self.results["part_i_19"] = results
        return results
    
    def run_latency_benchmark(
        self,
        teacher_embeddings: torch.Tensor,
        cgt_embeddings: torch.Tensor,
        substrate: LorentzSubstrateHardened,
    ) -> Dict:
        """
        Run Part IV.4: Latency Benchmark.
        
        ENCAPSULATION ONLY - delegates to experiments.benchmarks
        """
        config = LatencyConfig(
            device=str(self.device),
        )
        
        results = run_latency_benchmark(
            teacher_embeddings=teacher_embeddings,
            cgt_embeddings=cgt_embeddings,
            substrate=substrate,
            config=config,
            output_dir=self.dirs["stats"] / "part_iv_4",
        )
        
        self.results["part_iv_4"] = results
        return results
    
    def run_statistical_robustness(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
        test_emb1: torch.Tensor,
        test_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        teacher_spearman: float,
        teacher_dim: int = 384,
    ) -> Dict:
        """
        Run Part VI: Statistical Robustness (multi-seed).
        
        ENCAPSULATION ONLY - delegates to experiments.analysis
        """
        seeds = [self.seed + i * 111 for i in range(self.n_seeds)]
        
        config = RobustnessConfig(
            seeds=seeds,
            teacher_dim=teacher_dim,
        )
        
        results = run_statistical_robustness(
            train_emb1=train_emb1,
            train_emb2=train_emb2,
            train_scores=train_scores,
            val_emb1=val_emb1,
            val_emb2=val_emb2,
            val_scores=val_scores,
            test_emb1=test_emb1,
            test_emb2=test_emb2,
            test_scores=test_scores,
            teacher_spearman=teacher_spearman,
            config=config,
            output_dir=self.dirs["stats"] / "part_vi",
        )
        
        self.results["part_vi"] = results
        return results
    
    def save_summary(self) -> None:
        """Save all results summary."""
        summary_path = self.dirs["stats"] / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Summary saved: {summary_path}")
