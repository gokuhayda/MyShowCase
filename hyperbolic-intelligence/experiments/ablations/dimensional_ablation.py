# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.1b - Dimensional Ablation Study
=======================================

AUDIT COMPLIANCE:
- Uses ONLY hardened modules for CGT
- Uses ONLY PyTorch built-ins for Euclidean
- Reuses components from Part IV.1

PURPOSE:
Test CGT vs Euclidean across multiple dimensions to find the "crossover point"
where hyperbolic geometry starts providing advantage.

HYPOTHESIS:
- At high dimensions (32+), Euclidean space has sufficient volume
- At low dimensions (4-8), Euclidean "suffocates" and collapses
- Hyperbolic space maintains separation due to exponential volume growth

OUTPUT:
- Performance curve: Both methods vs dimension
- Crossover dimension identification
- LaTeX table for paper

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import from hardened modules
from cgt.utils.helpers import set_global_seed

# Import from Part IV.1
from .euclidean_ablation import (
    AblationConfig,
    train_euclidean_model,
    evaluate_euclidean_stsb,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionalAblationConfig:
    """Configuration for dimensional ablation study."""
    # Dimensions to test
    test_dimensions: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256])
    
    # Architecture
    teacher_dim: int = 384
    hidden_dim: int = 256
    
    # Training (reduced epochs for faster iteration)
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 15  # Reduced for ablation
    weight_decay: float = 1e-5
    
    # Loss weights
    lambda_contrastive: float = 1.0
    lambda_distill: float = 0.5
    temperature: float = 0.07
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_ablation_config(self, student_dim: int) -> AblationConfig:
        """Convert to AblationConfig with specific dimension."""
        return AblationConfig(
            teacher_dim=self.teacher_dim,
            student_dim=student_dim,
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
            seed=self.seed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                    CGT TRAINING AT SPECIFIC DIMENSION
# ═══════════════════════════════════════════════════════════════════════════════

def train_cgt_at_dimension(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    val_emb1: torch.Tensor,
    val_emb2: torch.Tensor,
    val_scores: torch.Tensor,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    config: AblationConfig,
) -> Dict[str, float]:
    """
    Train CGT model at a specific dimension and return metrics.
    
    Uses HARDENED modules only.
    """
    from cgt.models.cgt_hardened import CGTStudentHardened, RiemannianOptimizerWrapper
    from cgt.losses.losses_hardened import MultiObjectiveLoss
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    # Initialize model
    model = CGTStudentHardened(
        teacher_dim=config.teacher_dim,
        student_dim=config.student_dim,
        hidden_dim=config.hidden_dim,
        learnable_curvature=True,
        initial_curvature=1.0,
    ).to(device, dtype)
    
    model.init_homeostatic(n_anchors=16, alpha=0.1)
    
    # Loss and optimizer
    criterion = MultiObjectiveLoss(
        lambda_contrastive=config.lambda_contrastive,
        lambda_distill=config.lambda_distill,
        temperature=config.temperature,
    )
    
    base_optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer = RiemannianOptimizerWrapper(base_optimizer, model.substrate)
    
    # Data
    train_emb1 = train_emb1.to(device, dtype)
    train_emb2 = train_emb2.to(device, dtype)
    train_dataset = TensorDataset(train_emb1, train_emb2, train_scores.to(device, dtype))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training
    best_val_spearman = -1.0
    lorentz = model.substrate
    
    for epoch in range(config.num_epochs):
        model.train()
        for batch_emb1, batch_emb2, _ in train_loader:
            optimizer.zero_grad()
            
            student_emb1 = model(batch_emb1)
            student_emb2 = model(batch_emb2)
            
            student_emb = torch.cat([student_emb1, student_emb2], dim=0)
            teacher_emb = torch.cat([batch_emb1, batch_emb2], dim=0)
            
            losses = criterion(student_emb, teacher_emb, model, current_epoch=epoch)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_student1 = model(val_emb1.to(device, dtype))
            val_student2 = model(val_emb2.to(device, dtype))
            sims = lorentz.lorentz_similarity(val_student1, val_student2).squeeze()
            val_spearman, _ = spearmanr(sims.cpu().numpy(), val_scores.numpy())
            
            if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_student1 = model(test_emb1.to(device, dtype))
        test_student2 = model(test_emb2.to(device, dtype))
        sims = lorentz.lorentz_similarity(test_student1, test_student2).squeeze()
        test_spearman, _ = spearmanr(sims.cpu().numpy(), test_scores.numpy())
    
    return {
        'test_spearman': float(test_spearman) if not np.isnan(test_spearman) else 0.0,
        'best_val_spearman': float(best_val_spearman) if not np.isnan(best_val_spearman) else 0.0,
        'final_curvature': model.get_curvature(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    DIMENSIONAL ABLATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_dimensional_ablation(
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
    config: DimensionalAblationConfig,
    output_dir: Path,
) -> Dict:
    """
    Run dimensional ablation study.
    
    Tests CGT vs Euclidean across multiple dimensions to find crossover point.
    """
    print("\n" + "=" * 80)
    print("PART IV.1b - DIMENSIONAL ABLATION STUDY")
    print("Finding the Euclidean Breaking Point")
    print("=" * 80)
    
    set_global_seed(config.seed)
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'dimensional_results': [],
    }
    
    print(f"\nDimensions to test: {config.test_dimensions}")
    print(f"Epochs per model: {config.num_epochs}")
    
    # Test each dimension
    for dim in config.test_dimensions:
        print(f"\n{'='*60}")
        print(f"DIMENSION: {dim}")
        print(f"{'='*60}")
        
        ablation_config = config.to_ablation_config(dim)
        
        # Train CGT
        print(f"  Training CGT-{dim}...")
        set_global_seed(config.seed)
        cgt_metrics = train_cgt_at_dimension(
            train_emb1, train_emb2, train_scores,
            val_emb1, val_emb2, val_scores,
            test_emb1, test_emb2, test_scores,
            ablation_config,
        )
        
        # Train Euclidean
        print(f"  Training Euclidean-{dim}...")
        set_global_seed(config.seed)
        euc_model, _ = train_euclidean_model(
            train_emb1, train_emb2, train_scores,
            val_emb1, val_emb2, val_scores,
            ablation_config,
        )
        
        device = torch.device(config.device)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        
        euc_metrics = evaluate_euclidean_stsb(
            euc_model, test_emb1, test_emb2, test_scores, device, dtype
        )
        
        # Record results
        dim_result = {
            'dimension': dim,
            'cgt_spearman': cgt_metrics['test_spearman'],
            'euclidean_spearman': euc_metrics['spearman'],
            'cgt_retention': cgt_metrics['test_spearman'] / teacher_spearman * 100,
            'euclidean_retention': euc_metrics['spearman'] / teacher_spearman * 100,
            'advantage': cgt_metrics['test_spearman'] - euc_metrics['spearman'],
            'cgt_curvature': cgt_metrics['final_curvature'],
        }
        results['dimensional_results'].append(dim_result)
        
        print(f"  CGT-{dim}:       ρ = {cgt_metrics['test_spearman']:.4f}")
        print(f"  Euclidean-{dim}: ρ = {euc_metrics['spearman']:.4f}")
        print(f"  Advantage:      {dim_result['advantage']:+.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS: Find Crossover Point
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("ANALYSIS: CROSSOVER POINT")
    print("=" * 80)
    
    # Find dimension where CGT first beats Euclidean
    crossover_dim = None
    for result in results['dimensional_results']:
        if result['advantage'] > 0:
            crossover_dim = result['dimension']
            break
    
    results['crossover_dimension'] = crossover_dim
    
    if crossover_dim:
        print(f"\n✅ Crossover found at dimension {crossover_dim}")
        print(f"   CGT begins to outperform Euclidean at {crossover_dim}D")
    else:
        print("\n⚠️ No crossover found - Euclidean competitive at all tested dimensions")
    
    # Summary table
    print("\n" + "-" * 70)
    print("DIMENSIONAL PERFORMANCE TABLE")
    print("-" * 70)
    print(f"{'Dim':>6} {'CGT ρ':>10} {'Euc ρ':>10} {'Advantage':>12} {'Winner':>10}")
    print("-" * 70)
    
    for r in results['dimensional_results']:
        winner = "CGT" if r['advantage'] > 0 else "Euclidean"
        print(f"{r['dimension']:>6} {r['cgt_spearman']:>10.4f} {r['euclidean_spearman']:>10.4f} "
              f"{r['advantage']:>+12.4f} {winner:>10}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / 'dimensional_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV
    summary_df = pd.DataFrame(results['dimensional_results'])
    summary_df.to_csv(output_dir / 'dimensional_ablation_summary.csv', index=False)
    
    # LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{CGT vs Euclidean Performance Across Dimensions}",
        "\\label{tab:dimensional_ablation}",
        "\\begin{tabular}{rcccc}",
        "\\toprule",
        "Dimension & CGT $\\rho$ & Euclidean $\\rho$ & Advantage & Winner \\\\",
        "\\midrule",
    ]
    
    for r in results['dimensional_results']:
        winner = "CGT" if r['advantage'] > 0 else "Euclidean"
        latex_lines.append(
            f"{r['dimension']} & {r['cgt_spearman']:.4f} & {r['euclidean_spearman']:.4f} & "
            f"{r['advantage']:+.4f} & {winner} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with open(output_dir / 'dimensional_ablation_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.1b - DIMENSIONAL ABLATION STUDY")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings.")
    print("   Use run_dimensional_ablation() with your data.")
