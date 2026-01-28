# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.1c - Geometric Capacity Collapse Analysis
=================================================

AUDIT COMPLIANCE:
- Uses ONLY hardened modules for CGT
- Uses ONLY PyTorch built-ins for Euclidean
- Pure mathematical analysis of geometric capacity

PURPOSE:
Analyze WHY Euclidean collapses at low dimensions while hyperbolic doesn't.
This provides the theoretical grounding for CGT's advantage.

KEY METRICS:
1. Isotropy Score: How uniformly spread are embeddings?
2. Effective Rank: How many dimensions are actually used?
3. Pair Distance Distribution: Are pairs separable?
4. Volume Utilization: How much of the space is used?

HYPOTHESIS:
Euclidean space at d=2 has volume V = πr² (polynomial)
Hyperbolic space at d=2 has volume V ∝ e^r (exponential)

When compressing 50K+ sentences to d=2:
- Euclidean: Points collapse to center (low isotropy)
- Hyperbolic: Points spread exponentially (high isotropy)

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import entropy, spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cgt.utils.helpers import set_global_seed


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometricCapacityConfig:
    """Configuration for geometric capacity analysis."""
    # Dimensions to test (focus on low-d)
    test_dimensions: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    
    # Architecture
    teacher_dim: int = 384
    hidden_dim: int = 256
    
    # Training (reduced epochs)
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 10
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


# ═══════════════════════════════════════════════════════════════════════════════
#                    GEOMETRIC METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_isotropy_score(embeddings: np.ndarray) -> float:
    """
    Compute isotropy score: how uniformly spread are embeddings.
    
    Score of 1.0 = perfectly isotropic (uniform distribution)
    Score of 0.0 = all points collapsed to origin
    
    Method: Compare singular value distribution to uniform
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)
    
    # SVD
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    
    # Normalize singular values
    S_norm = S / S.sum()
    
    # Compute entropy vs max entropy
    max_entropy = np.log(len(S))
    actual_entropy = entropy(S_norm + 1e-10)
    
    isotropy = actual_entropy / max_entropy
    return float(isotropy)


def compute_effective_rank(embeddings: np.ndarray, threshold: float = 0.99) -> int:
    """
    Compute effective rank: how many dimensions capture 99% of variance.
    
    Low effective rank = embeddings collapsed to low-d subspace
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    
    # Cumulative variance explained
    var_explained = np.cumsum(S**2) / np.sum(S**2)
    
    # Find first k where cumulative >= threshold
    effective_rank = np.argmax(var_explained >= threshold) + 1
    return int(effective_rank)


def compute_pair_separation(
    emb1: np.ndarray, 
    emb2: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Analyze how well similar vs dissimilar pairs are separated.
    
    Returns:
        - separation_ratio: distance_dissimilar / distance_similar
        - overlap_score: how much distributions overlap (lower = better)
    """
    # Compute pairwise cosine similarities
    norm1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    norm2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(norm1 * norm2, axis=1)
    
    # Split by ground truth
    median_score = np.median(scores)
    similar_mask = scores >= median_score
    dissimilar_mask = scores < median_score
    
    # Distance distributions (1 - sim)
    similar_dists = 1 - sims[similar_mask]
    dissimilar_dists = 1 - sims[dissimilar_mask]
    
    # Separation ratio
    mean_similar = similar_dists.mean()
    mean_dissimilar = dissimilar_dists.mean()
    separation_ratio = mean_dissimilar / (mean_similar + 1e-8)
    
    # Overlap score via histogram intersection
    hist_similar, bins = np.histogram(similar_dists, bins=n_bins, range=(0, 2), density=True)
    hist_dissimilar, _ = np.histogram(dissimilar_dists, bins=bins, density=True)
    
    overlap = np.minimum(hist_similar, hist_dissimilar).sum() / n_bins
    
    return {
        'separation_ratio': float(separation_ratio),
        'overlap_score': float(overlap),
        'mean_similar_dist': float(mean_similar),
        'mean_dissimilar_dist': float(mean_dissimilar),
    }


def compute_volume_utilization(embeddings: np.ndarray) -> float:
    """
    Estimate volume utilization: fraction of space used.
    
    Method: Compare to uniform distribution in hypersphere
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit_emb = embeddings / (norms + 1e-8)
    
    # Compute pairwise distances on sphere
    pairwise_dists = pdist(unit_emb, metric='cosine')
    
    # Expected distance for uniform distribution on d-sphere
    d = embeddings.shape[1]
    expected_uniform_dist = 1.0  # Approximate for high-d
    
    actual_mean_dist = pairwise_dists.mean()
    
    # Volume utilization: ratio of actual spread to expected
    volume_util = actual_mean_dist / expected_uniform_dist
    
    return float(np.clip(volume_util, 0, 1))


def compute_hyperbolic_metrics(
    embeddings: torch.Tensor,
    lorentz_substrate,
) -> Dict[str, float]:
    """
    Compute hyperbolic-specific metrics.
    
    Args:
        embeddings: Lorentz embeddings [N, D+1]
        lorentz_substrate: LorentzSubstrate instance
    """
    emb_np = embeddings.cpu().numpy()
    
    # Time components
    time_coord = emb_np[:, 0]
    spatial_coord = emb_np[:, 1:]
    
    # Hyperbolic norms (geodesic distance from origin)
    # For Lorentz: norm² = x₀² - ||x||² should be 1/c
    c = float(lorentz_substrate.get_curvature())
    spatial_norms = np.linalg.norm(spatial_coord, axis=1)
    hyperbolic_norms = np.arccosh(np.clip(time_coord * np.sqrt(c), 1.0, None)) / np.sqrt(c)
    
    return {
        'mean_hyperbolic_norm': float(hyperbolic_norms.mean()),
        'std_hyperbolic_norm': float(hyperbolic_norms.std()),
        'mean_time_coord': float(time_coord.mean()),
        'mean_spatial_norm': float(spatial_norms.mean()),
        'curvature': float(c),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    TRAIN AND ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_analyze_cgt(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    student_dim: int,
    config: GeometricCapacityConfig,
) -> Dict:
    """Train CGT and compute geometric capacity metrics."""
    from cgt.models.cgt_hardened import CGTStudentHardened, RiemannianOptimizerWrapper
    from cgt.losses.losses_hardened import MultiObjectiveLoss
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    set_global_seed(config.seed)
    
    # Initialize model
    model = CGTStudentHardened(
        teacher_dim=config.teacher_dim,
        student_dim=student_dim,
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
    
    # Analyze
    model.eval()
    with torch.no_grad():
        test_student1 = model(test_emb1.to(device, dtype))
        test_student2 = model(test_emb2.to(device, dtype))
        
        # Spearman
        lorentz = model.substrate
        sims = lorentz.lorentz_similarity(test_student1, test_student2).squeeze()
        test_spearman, _ = spearmanr(sims.cpu().numpy(), test_scores.numpy())
        
        # Geometric metrics (spatial components)
        spatial1 = test_student1[:, 1:].cpu().numpy()
        spatial2 = test_student2[:, 1:].cpu().numpy()
        all_spatial = np.concatenate([spatial1, spatial2], axis=0)
        
    metrics = {
        'spearman': float(test_spearman),
        'isotropy': compute_isotropy_score(all_spatial),
        'effective_rank': compute_effective_rank(all_spatial),
        'volume_utilization': compute_volume_utilization(all_spatial),
        'pair_separation': compute_pair_separation(spatial1, spatial2, test_scores.numpy()),
        'hyperbolic': compute_hyperbolic_metrics(test_student1, lorentz),
    }
    
    return metrics


def train_and_analyze_euclidean(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    student_dim: int,
    config: GeometricCapacityConfig,
) -> Dict:
    """Train Euclidean and compute geometric capacity metrics."""
    from .euclidean_ablation import train_euclidean_model, evaluate_euclidean_stsb, AblationConfig
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    
    set_global_seed(config.seed)
    
    ablation_config = AblationConfig(
        teacher_dim=config.teacher_dim,
        student_dim=student_dim,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        lambda_contrastive=config.lambda_contrastive,
        lambda_distill=config.lambda_distill,
        temperature=config.temperature,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
    )
    
    # Train Euclidean
    model, _ = train_euclidean_model(
        train_emb1, train_emb2, train_scores,
        test_emb1, test_emb2, test_scores,  # Use test as val for simplicity
        ablation_config,
    )
    
    # Analyze
    model.eval()
    with torch.no_grad():
        test_student1 = model(test_emb1.to(device, dtype))
        test_student2 = model(test_emb2.to(device, dtype))
        
        # Normalize and compute similarity
        norm1 = F.normalize(test_student1, dim=-1)
        norm2 = F.normalize(test_student2, dim=-1)
        sims = (norm1 * norm2).sum(dim=-1)
        test_spearman, _ = spearmanr(sims.cpu().numpy(), test_scores.numpy())
        
        all_emb = torch.cat([test_student1, test_student2], dim=0).cpu().numpy()
        
    metrics = {
        'spearman': float(test_spearman),
        'isotropy': compute_isotropy_score(all_emb),
        'effective_rank': compute_effective_rank(all_emb),
        'volume_utilization': compute_volume_utilization(all_emb),
        'pair_separation': compute_pair_separation(
            test_student1.cpu().numpy(),
            test_student2.cpu().numpy(),
            test_scores.numpy()
        ),
    }
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#                    VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_capacity_analysis(
    results: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot geometric capacity analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    dimensions = results['dimensions']
    cgt_metrics = results['cgt']
    euc_metrics = results['euclidean']
    
    # 1. Spearman comparison
    ax1 = axes[0, 0]
    ax1.plot(dimensions, [m['spearman'] for m in cgt_metrics], 'o-', label='CGT', linewidth=2, markersize=8)
    ax1.plot(dimensions, [m['spearman'] for m in euc_metrics], 's--', label='Euclidean', linewidth=2, markersize=8)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Spearman ρ')
    ax1.set_title('Performance vs Dimension', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Isotropy comparison
    ax2 = axes[0, 1]
    ax2.plot(dimensions, [m['isotropy'] for m in cgt_metrics], 'o-', label='CGT', linewidth=2)
    ax2.plot(dimensions, [m['isotropy'] for m in euc_metrics], 's--', label='Euclidean', linewidth=2)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Isotropy Score')
    ax2.set_title('Isotropy (Higher = Better)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Effective Rank
    ax3 = axes[0, 2]
    ax3.plot(dimensions, [m['effective_rank'] for m in cgt_metrics], 'o-', label='CGT', linewidth=2)
    ax3.plot(dimensions, [m['effective_rank'] for m in euc_metrics], 's--', label='Euclidean', linewidth=2)
    ax3.plot(dimensions, dimensions, ':', color='gray', label='Max Rank', alpha=0.7)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Effective Rank')
    ax3.set_title('Effective Rank (Higher = Better)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Volume Utilization
    ax4 = axes[1, 0]
    ax4.plot(dimensions, [m['volume_utilization'] for m in cgt_metrics], 'o-', label='CGT', linewidth=2)
    ax4.plot(dimensions, [m['volume_utilization'] for m in euc_metrics], 's--', label='Euclidean', linewidth=2)
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Volume Utilization')
    ax4.set_title('Space Utilization (Higher = Better)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Separation Ratio
    ax5 = axes[1, 1]
    ax5.plot(dimensions, [m['pair_separation']['separation_ratio'] for m in cgt_metrics], 'o-', label='CGT', linewidth=2)
    ax5.plot(dimensions, [m['pair_separation']['separation_ratio'] for m in euc_metrics], 's--', label='Euclidean', linewidth=2)
    ax5.set_xlabel('Dimension')
    ax5.set_ylabel('Separation Ratio')
    ax5.set_title('Pair Separation (Higher = Better)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Bar Chart at d=2
    ax6 = axes[1, 2]
    metrics_names = ['Spearman', 'Isotropy', 'Volume', 'Separation']
    cgt_d2 = cgt_metrics[0]  # d=2
    euc_d2 = euc_metrics[0]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    cgt_vals = [cgt_d2['spearman'], cgt_d2['isotropy'], cgt_d2['volume_utilization'], 
                cgt_d2['pair_separation']['separation_ratio']]
    euc_vals = [euc_d2['spearman'], euc_d2['isotropy'], euc_d2['volume_utilization'],
                euc_d2['pair_separation']['separation_ratio']]
    
    ax6.bar(x - width/2, cgt_vals, width, label='CGT', alpha=0.8)
    ax6.bar(x + width/2, euc_vals, width, label='Euclidean', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_names)
    ax6.set_ylabel('Score')
    ax6.set_title('Metrics at d=2: CGT vs Euclidean', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved figure to: {output_path}")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#                    RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_geometric_capacity_analysis(
    train_emb1: torch.Tensor,
    train_emb2: torch.Tensor,
    train_scores: torch.Tensor,
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    teacher_spearman: float,
    config: GeometricCapacityConfig,
    output_dir: Path,
) -> Dict:
    """
    Run geometric capacity collapse analysis.
    
    Compare CGT vs Euclidean geometric properties across dimensions.
    """
    print("\n" + "=" * 80)
    print("PART IV.1c - GEOMETRIC CAPACITY COLLAPSE ANALYSIS")
    print("Why Hyperbolic Geometry Prevents Embedding Collapse")
    print("=" * 80)
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'dimensions': config.test_dimensions,
        'cgt': [],
        'euclidean': [],
    }
    
    for dim in config.test_dimensions:
        print(f"\n{'='*60}")
        print(f"ANALYZING DIMENSION: {dim}")
        print(f"{'='*60}")
        
        # CGT
        print(f"  Training & analyzing CGT-{dim}...")
        cgt_metrics = train_and_analyze_cgt(
            train_emb1, train_emb2, train_scores,
            test_emb1, test_emb2, test_scores,
            student_dim=dim,
            config=config,
        )
        results['cgt'].append(cgt_metrics)
        
        # Euclidean
        print(f"  Training & analyzing Euclidean-{dim}...")
        euc_metrics = train_and_analyze_euclidean(
            train_emb1, train_emb2, train_scores,
            test_emb1, test_emb2, test_scores,
            student_dim=dim,
            config=config,
        )
        results['euclidean'].append(euc_metrics)
        
        # Print comparison
        print(f"\n  Results at d={dim}:")
        print(f"    {'Metric':<20} {'CGT':>10} {'Euclidean':>10} {'Advantage':>10}")
        print(f"    {'-'*50}")
        print(f"    {'Spearman ρ':<20} {cgt_metrics['spearman']:>10.4f} {euc_metrics['spearman']:>10.4f} {cgt_metrics['spearman']-euc_metrics['spearman']:>+10.4f}")
        print(f"    {'Isotropy':<20} {cgt_metrics['isotropy']:>10.4f} {euc_metrics['isotropy']:>10.4f} {cgt_metrics['isotropy']-euc_metrics['isotropy']:>+10.4f}")
        print(f"    {'Effective Rank':<20} {cgt_metrics['effective_rank']:>10} {euc_metrics['effective_rank']:>10}")
        print(f"    {'Volume Util.':<20} {cgt_metrics['volume_utilization']:>10.4f} {euc_metrics['volume_utilization']:>10.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("GEOMETRIC CAPACITY COLLAPSE ANALYSIS")
    print("=" * 80)
    
    # Find where collapse happens
    collapse_dim = None
    for i, dim in enumerate(config.test_dimensions):
        cgt_iso = results['cgt'][i]['isotropy']
        euc_iso = results['euclidean'][i]['isotropy']
        
        if euc_iso < 0.5 and cgt_iso > 0.5:  # Euclidean collapsed, CGT didn't
            collapse_dim = dim
            break
    
    results['collapse_analysis'] = {
        'euclidean_collapse_dim': collapse_dim,
        'hypothesis_confirmed': collapse_dim is not None,
    }
    
    if collapse_dim:
        print(f"\n✅ HYPOTHESIS CONFIRMED")
        print(f"   Euclidean shows collapse at d≤{collapse_dim}")
        print(f"   CGT maintains isotropy through hyperbolic geometry")
    else:
        print(f"\n⚠️ No clear collapse point found in tested dimensions")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / 'geometric_capacity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Visualization
    fig = plot_capacity_analysis(results, output_dir / 'geometric_capacity_analysis.png')
    fig.savefig(output_dir / 'geometric_capacity_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # CSV summary
    rows = []
    for i, dim in enumerate(config.test_dimensions):
        rows.append({
            'dimension': dim,
            'cgt_spearman': results['cgt'][i]['spearman'],
            'euc_spearman': results['euclidean'][i]['spearman'],
            'cgt_isotropy': results['cgt'][i]['isotropy'],
            'euc_isotropy': results['euclidean'][i]['isotropy'],
            'cgt_effective_rank': results['cgt'][i]['effective_rank'],
            'euc_effective_rank': results['euclidean'][i]['effective_rank'],
            'cgt_volume': results['cgt'][i]['volume_utilization'],
            'euc_volume': results['euclidean'][i]['volume_utilization'],
        })
    
    pd.DataFrame(rows).to_csv(output_dir / 'geometric_capacity_summary.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.1c - GEOMETRIC CAPACITY COLLAPSE ANALYSIS")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings.")
    print("   Use run_geometric_capacity_analysis() with your data.")
