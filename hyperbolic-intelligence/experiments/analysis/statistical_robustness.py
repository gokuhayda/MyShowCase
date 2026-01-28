# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part VI - Statistical Robustness (Seed Sensitivity Analysis)
============================================================

AUDIT COMPLIANCE:
- Uses ONLY hardened modules for CGT training
- Uses ONLY PyTorch built-ins for Euclidean baseline
- No formula derivation

PURPOSE:
Ensure CGT gains are not due to lucky initialization by running
experiments across multiple random seeds.

This neutralizes the classic reviewer attack:
"This could just be initialization noise."

OUTPUT:
- Mean ± Std Dev for CGT and Euclidean
- Statistical significance test (Wilcoxon signed-rank)
- Confidence intervals

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import from hardened modules
from cgt.utils.helpers import set_global_seed

# Import ablation components (from ablations subpackage, not analysis)
from ablations.euclidean_ablation import (
    AblationConfig,
    train_cgt_model,
    train_euclidean_model,
    evaluate_euclidean_stsb,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessConfig:
    """Configuration for statistical robustness study."""
    # Seeds for multiple runs
    seeds: List[int] = None  # Will default to [42, 123, 456]
    
    # Training config (inherited from AblationConfig)
    teacher_dim: int = 384
    student_dim: int = 32
    hidden_dim: int = 256
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 25
    weight_decay: float = 1e-5
    lambda_contrastive: float = 1.0
    lambda_distill: float = 0.5
    temperature: float = 0.07
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]  # Minimum 3 seeds for statistical validity
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_ablation_config(self, seed: int) -> AblationConfig:
        """Convert to AblationConfig with specific seed."""
        return AblationConfig(
            teacher_dim=self.teacher_dim,
            student_dim=self.student_dim,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            lambda_contrastive=self.lambda_contrastive,
            lambda_distill=self.lambda_distill,
            temperature=self.temperature,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                    MULTI-SEED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical_robustness(
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
    config: RobustnessConfig,
    output_dir: Path,
) -> Dict:
    """
    Run statistical robustness study with multiple seeds.
    
    Trains both CGT and Euclidean baselines across multiple random seeds
    and computes statistical significance of the difference.
    
    Args:
        train_emb1, train_emb2, train_scores: Training data
        val_emb1, val_emb2, val_scores: Validation data
        test_emb1, test_emb2, test_scores: Test data
        teacher_spearman: Teacher model baseline
        config: Robustness configuration
        output_dir: Output directory for results
    
    Returns:
        Dictionary with robustness results
    """
    print("\n" + "=" * 80)
    print("PART VI - STATISTICAL ROBUSTNESS (SEED SENSITIVITY)")
    print("=" * 80)
    print(f"\nRunning {len(config.seeds)} seeds: {config.seeds}")
    
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'seed_results': [],
    }
    
    cgt_spearmans = []
    euc_spearmans = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUN EXPERIMENTS FOR EACH SEED
    # ═══════════════════════════════════════════════════════════════════════
    
    for i, seed in enumerate(config.seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(config.seeds)}: {seed}")
        print(f"{'='*60}")
        
        # Set global seed
        set_global_seed(seed)
        
        ablation_config = config.to_ablation_config(seed)
        
        # Train CGT
        print(f"\n[Seed {seed}] Training CGT...")
        cgt_model, cgt_history = train_cgt_model(
            train_emb1, train_emb2, train_scores,
            val_emb1, val_emb2, val_scores,
            ablation_config,
        )
        
        # Evaluate CGT
        cgt_model.eval()
        lorentz = cgt_model.substrate
        with torch.no_grad():
            cgt_test_emb1 = cgt_model(test_emb1.to(device, dtype))
            cgt_test_emb2 = cgt_model(test_emb2.to(device, dtype))
            from scipy.stats import spearmanr
            cgt_sims = lorentz.lorentz_similarity(cgt_test_emb1, cgt_test_emb2).squeeze()
            cgt_spearman, _ = spearmanr(cgt_sims.cpu().numpy(), test_scores.numpy())
        
        cgt_spearmans.append(float(cgt_spearman))
        
        # Train Euclidean
        print(f"[Seed {seed}] Training Euclidean...")
        set_global_seed(seed)  # Reset seed for fair comparison
        
        euc_model, euc_history = train_euclidean_model(
            train_emb1, train_emb2, train_scores,
            val_emb1, val_emb2, val_scores,
            ablation_config,
        )
        
        # Evaluate Euclidean
        euc_metrics = evaluate_euclidean_stsb(
            euc_model, test_emb1, test_emb2, test_scores, device, dtype
        )
        euc_spearmans.append(euc_metrics['spearman'])
        
        # Record seed results
        seed_result = {
            'seed': seed,
            'cgt_spearman': float(cgt_spearman),
            'euclidean_spearman': euc_metrics['spearman'],
            'cgt_retention': float(cgt_spearman / teacher_spearman * 100),
            'euclidean_retention': euc_metrics['spearman'] / teacher_spearman * 100,
            'cgt_advantage': float(cgt_spearman) - euc_metrics['spearman'],
        }
        results['seed_results'].append(seed_result)
        
        print(f"[Seed {seed}] CGT: ρ={cgt_spearman:.4f}, Euclidean: ρ={euc_metrics['spearman']:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    cgt_spearmans = np.array(cgt_spearmans)
    euc_spearmans = np.array(euc_spearmans)
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Descriptive statistics
    cgt_mean, cgt_std = cgt_spearmans.mean(), cgt_spearmans.std()
    euc_mean, euc_std = euc_spearmans.mean(), euc_spearmans.std()
    
    print(f"\nCGT:       ρ = {cgt_mean:.4f} ± {cgt_std:.4f}")
    print(f"Euclidean: ρ = {euc_mean:.4f} ± {euc_std:.4f}")
    print(f"Advantage: {(cgt_mean - euc_mean):.4f} ± {np.sqrt(cgt_std**2 + euc_std**2):.4f}")
    
    # Statistical tests
    if len(config.seeds) >= 3:
        # Paired t-test
        t_stat, t_pvalue = ttest_rel(cgt_spearmans, euc_spearmans)
        print(f"\nPaired t-test: t = {t_stat:.4f}, p = {t_pvalue:.6f}")
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = wilcoxon(cgt_spearmans, euc_spearmans)
            print(f"Wilcoxon test: W = {w_stat:.4f}, p = {w_pvalue:.6f}")
        except ValueError:
            # Wilcoxon requires non-zero differences
            w_stat, w_pvalue = None, None
            print("Wilcoxon test: N/A (requires non-zero differences)")
        
        # 95% Confidence Interval for mean difference
        diff = cgt_spearmans - euc_spearmans
        diff_mean = diff.mean()
        diff_se = diff.std() / np.sqrt(len(diff))
        ci_low = diff_mean - 1.96 * diff_se
        ci_high = diff_mean + 1.96 * diff_se
        print(f"\n95% CI for CGT-Euclidean difference: [{ci_low:.4f}, {ci_high:.4f}]")
        
        results['statistics'] = {
            'cgt_mean': float(cgt_mean),
            'cgt_std': float(cgt_std),
            'euclidean_mean': float(euc_mean),
            'euclidean_std': float(euc_std),
            'mean_advantage': float(diff_mean),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'wilcoxon_statistic': float(w_stat) if w_stat else None,
            'wilcoxon_pvalue': float(w_pvalue) if w_pvalue else None,
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
        }
        
        # Significance check
        alpha = 0.05
        significant = t_pvalue < alpha
        
        print("\n" + "-" * 60)
        if significant:
            print(f"✅ SIGNIFICANT at α = {alpha}: CGT advantage is real")
            print("   The difference is NOT due to random initialization.")
        else:
            print(f"⚠️ NOT SIGNIFICANT at α = {alpha}")
            print("   Cannot definitively rule out initialization noise.")
        
        results['significant'] = significant
        results['alpha'] = alpha
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'statistical_robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV summary
    summary_df = pd.DataFrame(results['seed_results'])
    summary_df.to_csv(output_dir / 'statistical_robustness_by_seed.csv', index=False)
    
    # Aggregated summary
    agg_df = pd.DataFrame([
        {
            'method': 'Teacher',
            'mean_spearman': teacher_spearman,
            'std_spearman': 0.0,
            'mean_retention': 100.0,
        },
        {
            'method': 'CGT',
            'mean_spearman': cgt_mean,
            'std_spearman': cgt_std,
            'mean_retention': cgt_mean / teacher_spearman * 100,
        },
        {
            'method': 'Euclidean',
            'mean_spearman': euc_mean,
            'std_spearman': euc_std,
            'mean_retention': euc_mean / teacher_spearman * 100,
        },
    ])
    agg_df.to_csv(output_dir / 'statistical_robustness_aggregated.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART VI - STATISTICAL ROBUSTNESS (SEED SENSITIVITY)")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings.")
    print("   Use run_statistical_robustness() with your data.")
