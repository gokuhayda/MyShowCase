# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.2 - MRL (Matryoshka Representation Learning) Comparison
===============================================================

AUDIT COMPLIANCE:
- NO geometry derivation
- Truncation is standard Python slicing
- Comparison uses hardened modules for CGT

PURPOSE:
Compare CGT against Matryoshka Representation Learning truncation baseline.

MRL EXPLANATION:
Matryoshka models are trained such that truncating to the first d dimensions
preserves semantic quality. For non-MRL models (like all-MiniLM-L6-v2),
truncation degrades quality significantly.

FAIR METHODOLOGY:
- Same teacher model
- Same target dimension
- Same evaluation metric (STS-B Spearman)

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MRLConfig:
    """Configuration for MRL comparison."""
    teacher_dim: int = 384
    target_dims: List[int] = None  # Will default to [32, 64, 128, 256]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.target_dims is None:
            self.target_dims = [32, 64, 128, 256]
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
#                    MRL TRUNCATION BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class MRLTruncation:
    """
    Matryoshka Representation Learning baseline.
    
    OPERATION:
    Simply truncates embeddings to the first d dimensions.
    This is equivalent to projecting onto the first d principal components
    if the model was trained with MRL objective.
    
    For non-MRL models, truncation causes significant quality degradation.
    """
    
    def __init__(self, target_dim: int):
        """
        Initialize MRL truncation baseline.
        
        Args:
            target_dim: Number of dimensions to keep
        """
        self.target_dim = target_dim
    
    def truncate(
        self,
        embeddings: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Truncate embeddings to first d dimensions.
        
        Args:
            embeddings: Full embeddings [N, D]
            normalize: Whether to L2-normalize after truncation
        
        Returns:
            Truncated embeddings [N, target_dim]
        """
        # Simple slicing - NOT geometry derivation
        truncated = embeddings[..., :self.target_dim]
        
        if normalize:
            # L2 normalization (PyTorch built-in)
            truncated = F.normalize(truncated, dim=-1)
        
        return truncated
    
    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity for truncated embeddings.
        
        Args:
            x: First set of embeddings [N, D]
            y: Second set of embeddings [N, D]
        
        Returns:
            Cosine similarities [N]
        """
        # Truncate
        x_trunc = self.truncate(x)
        y_trunc = self.truncate(y)
        
        # Cosine similarity (PyTorch built-in)
        return F.cosine_similarity(x_trunc, y_trunc, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#                    EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_mrl_truncation(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    target_dim: int,
) -> Dict[str, float]:
    """
    Evaluate MRL truncation on STS-B.
    
    Args:
        emb1: First sentence embeddings [N, D]
        emb2: Second sentence embeddings [N, D]
        scores: Ground truth similarity scores [N]
        target_dim: Target dimension for truncation
    
    Returns:
        Dictionary with evaluation metrics
    """
    mrl = MRLTruncation(target_dim)
    
    # Compute truncated similarities
    similarities = mrl.similarity(emb1, emb2)
    
    # Convert to numpy
    sims_np = similarities.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Spearman correlation
    spearman, p_value = spearmanr(sims_np, scores_np)
    
    return {
        'spearman': float(spearman) if not np.isnan(spearman) else 0.0,
        'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
        'target_dim': target_dim,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                    MRL COMPARISON RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_mrl_comparison(
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    teacher_spearman: float,
    cgt_spearman: float,
    config: MRLConfig,
    output_dir: Path,
) -> Dict:
    """
    Run MRL truncation comparison.
    
    Compares CGT against MRL truncation at various dimensions.
    
    Args:
        test_emb1: First sentence teacher embeddings [N, D]
        test_emb2: Second sentence teacher embeddings [N, D]
        test_scores: Ground truth similarity scores [N]
        teacher_spearman: Teacher model Spearman (baseline)
        cgt_spearman: CGT model Spearman (from Part IV.1 or Part I)
        config: MRL configuration
        output_dir: Directory for saving results
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 70)
    print("PART IV.2 - MRL (MATRYOSHKA) COMPARISON")
    print("=" * 70)
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'cgt_spearman': cgt_spearman,
        'mrl_results': [],
    }
    
    # Test MRL at various dimensions
    print("\nEvaluating MRL truncation at multiple dimensions...")
    print("-" * 50)
    
    for target_dim in config.target_dims:
        mrl_metrics = evaluate_mrl_truncation(
            test_emb1, test_emb2, test_scores, target_dim
        )
        
        mrl_metrics['retention'] = mrl_metrics['spearman'] / teacher_spearman * 100
        mrl_metrics['storage_bytes'] = target_dim * 4  # float32
        
        results['mrl_results'].append(mrl_metrics)
        
        print(f"  MRL-{target_dim:3d}: ρ={mrl_metrics['spearman']:.4f} "
              f"({mrl_metrics['retention']:.1f}% retention, "
              f"{mrl_metrics['storage_bytes']} bytes)")
    
    # Find MRL at CGT's dimension (typically 32)
    cgt_dim = 32  # Standard CGT dimension
    mrl_at_cgt_dim = next(
        (r for r in results['mrl_results'] if r['target_dim'] == cgt_dim),
        None
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: CGT vs MRL")
    print("=" * 70)
    
    print(f"\nTeacher (384D): ρ = {teacher_spearman:.4f}")
    print(f"CGT (32D):      ρ = {cgt_spearman:.4f} ({cgt_spearman/teacher_spearman*100:.1f}%)")
    
    if mrl_at_cgt_dim:
        mrl_32_spearman = mrl_at_cgt_dim['spearman']
        advantage = cgt_spearman - mrl_32_spearman
        
        print(f"MRL (32D):      ρ = {mrl_32_spearman:.4f} ({mrl_32_spearman/teacher_spearman*100:.1f}%)")
        print(f"\nCGT Advantage:  {advantage:+.4f}")
        
        results['comparison'] = {
            'cgt_vs_mrl_32_advantage': advantage,
            'cgt_wins': advantage > 0,
        }
        
        if advantage > 0:
            print("\n✅ CGT > MRL-32: CGT provides better compression quality")
        else:
            print("\n⚠️ CGT ≤ MRL-32: MRL truncation is competitive")
    
    # Find minimum MRL dimension that matches CGT performance
    for mrl_result in sorted(results['mrl_results'], key=lambda x: x['target_dim']):
        if mrl_result['spearman'] >= cgt_spearman:
            print(f"\n📊 MRL needs {mrl_result['target_dim']}D to match CGT's 32D performance")
            results['mrl_dim_to_match_cgt'] = mrl_result['target_dim']
            break
    else:
        print("\n📊 MRL cannot match CGT performance even at max dimension tested")
        results['mrl_dim_to_match_cgt'] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'mrl_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV summary
    rows = [
        {'method': 'Teacher', 'dim': 384, 'spearman': teacher_spearman, 
         'retention': 100.0, 'storage_bytes': 384 * 4},
        {'method': 'CGT', 'dim': 32, 'spearman': cgt_spearman,
         'retention': cgt_spearman / teacher_spearman * 100, 'storage_bytes': 33 * 4},
    ]
    for mrl_result in results['mrl_results']:
        rows.append({
            'method': f'MRL-{mrl_result["target_dim"]}',
            'dim': mrl_result['target_dim'],
            'spearman': mrl_result['spearman'],
            'retention': mrl_result['retention'],
            'storage_bytes': mrl_result['storage_bytes'],
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / 'mrl_comparison_summary.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.2 - MRL (MATRYOSHKA) COMPARISON")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings and CGT results.")
    print("   Use run_mrl_comparison() with your data.")
