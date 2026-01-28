# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.3 - Binary Quantization Comparison (BQ-768 vs CGT-32)
============================================================

AUDIT COMPLIANCE:
- Binary quantization uses numpy.sign() (standard library)
- Hamming distance uses numpy operations (standard library)
- CGT uses lorentz_similarity from cgt.geometry.lorentz_hardened
- No formula derivation

PURPOSE:
Compare CGT-32 against Binary Quantization of full 768D embeddings.

Binary Quantization (BQ) converts each dimension to a single bit:
- Storage: 768 bits = 96 bytes (vs CGT: 33×4 = 132 bytes)
- Similarity: Hamming distance or XNOR popcount

TRADE-OFF ANALYSIS:
- BQ-768: 96 bytes, fast Hamming distance, good for retrieval
- CGT-32: 132 bytes (float32) or 66 bytes (float16), requires float ops

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
#                    BINARY QUANTIZATION (numpy - standard library)
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryQuantization:
    """
    Binary Quantization for embeddings.
    
    Converts floating-point embeddings to binary by thresholding.
    Similarity is computed via normalized Hamming distance.
    
    OPERATION:
    - sign(x) returns -1 or +1 for each dimension
    - This is standard numpy, NOT geometry derivation
    
    Storage comparison (per embedding):
    - FP32-768: 3072 bytes
    - FP32-32:  128 bytes (CGT spatial)
    - BQ-768:   96 bytes (768 bits)
    - BQ-384:   48 bytes (384 bits)
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize binary quantizer.
        
        Args:
            threshold: Threshold for binarization (default: 0, sign function)
        """
        self.threshold = threshold
    
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize embeddings to binary (-1 or +1).
        
        Args:
            embeddings: Float embeddings [N, D]
        
        Returns:
            Binary embeddings (-1 or +1) [N, D]
        """
        # numpy.sign is standard library - NOT geometry derivation
        return np.sign(embeddings - self.threshold)
    
    def hamming_similarity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Hamming similarity between binary vectors.
        
        For binary vectors with values in {-1, +1}:
        - Inner product: sum(x * y)
        - Range: [-D, D]
        - Normalized similarity: (inner + D) / (2*D) ∈ [0, 1]
        
        Args:
            x: Binary embeddings [N, D]
            y: Binary embeddings [N, D] or [M, D]
        
        Returns:
            Hamming similarity ∈ [0, 1]
        """
        D = x.shape[-1]
        
        # Ensure binary
        x_bin = np.sign(x)
        y_bin = np.sign(y)
        
        # Inner product for matching dimensions
        if x_bin.shape == y_bin.shape:
            inner = np.sum(x_bin * y_bin, axis=-1)
        else:
            # Pairwise comparison
            inner = np.dot(x_bin, y_bin.T)
        
        # Normalize to [0, 1]
        # D matches → inner = D → similarity = 1
        # D mismatches → inner = -D → similarity = 0
        similarity = (inner + D) / (2 * D)
        
        return similarity
    
    @staticmethod
    def storage_bytes(dim: int) -> int:
        """Calculate storage in bytes for binary embeddings."""
        return dim // 8  # 1 bit per dimension


# ═══════════════════════════════════════════════════════════════════════════════
#                    COMPARISON RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BQComparisonConfig:
    """Configuration for BQ comparison."""
    bq_dimensions: list = None  # Dimensions to test for BQ
    cgt_dim: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    
    def __post_init__(self):
        if self.bq_dimensions is None:
            self.bq_dimensions = [768, 384, 256, 128]
    
    def to_dict(self) -> dict:
        return asdict(self)


def run_bq_comparison(
    test_emb1: torch.Tensor,
    test_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    cgt_emb1: torch.Tensor,  # Pre-computed CGT embeddings
    cgt_emb2: torch.Tensor,
    cgt_substrate,  # LorentzSubstrateHardened for similarity computation
    teacher_spearman: float,
    cgt_spearman: float,
    config: BQComparisonConfig,
    output_dir: Path,
) -> Dict:
    """
    Run Binary Quantization comparison.
    
    Compares CGT against BQ at various dimensions.
    
    Args:
        test_emb1, test_emb2: Teacher test embeddings [N, D]
        test_scores: Ground truth similarity scores [N]
        cgt_emb1, cgt_emb2: Pre-computed CGT embeddings [N, D+1]
        cgt_substrate: Lorentz substrate for CGT similarity
        teacher_spearman: Teacher baseline
        cgt_spearman: CGT baseline (pre-computed)
        config: Configuration
        output_dir: Output directory
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 80)
    print("PART IV.3 - BINARY QUANTIZATION COMPARISON")
    print("=" * 80)
    
    # Convert to numpy
    test_emb1_np = test_emb1.cpu().numpy() if isinstance(test_emb1, torch.Tensor) else test_emb1
    test_emb2_np = test_emb2.cpu().numpy() if isinstance(test_emb2, torch.Tensor) else test_emb2
    test_scores_np = test_scores.cpu().numpy() if isinstance(test_scores, torch.Tensor) else test_scores
    
    teacher_dim = test_emb1_np.shape[1]
    
    results = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'teacher_spearman': teacher_spearman,
        'cgt_spearman': cgt_spearman,
        'methods': [],
    }
    
    bq = BinaryQuantization()
    
    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATE BQ AT VARIOUS DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\nEvaluating Binary Quantization...")
    print("-" * 60)
    
    for bq_dim in config.bq_dimensions:
        if bq_dim > teacher_dim:
            print(f"  Skipping BQ-{bq_dim}: exceeds teacher dimension ({teacher_dim})")
            continue
        
        # Truncate to target dimension (if needed) then binarize
        emb1_trunc = test_emb1_np[:, :bq_dim]
        emb2_trunc = test_emb2_np[:, :bq_dim]
        
        # Quantize
        emb1_bin = bq.quantize(emb1_trunc)
        emb2_bin = bq.quantize(emb2_trunc)
        
        # Compute similarity
        bq_sims = bq.hamming_similarity(emb1_bin, emb2_bin)
        
        # Evaluate
        bq_spearman, _ = spearmanr(bq_sims, test_scores_np)
        bq_spearman = float(bq_spearman) if not np.isnan(bq_spearman) else 0.0
        
        storage_bytes = bq_dim // 8
        
        method_result = {
            'method': f'BQ-{bq_dim}',
            'dim': bq_dim,
            'spearman': bq_spearman,
            'retention': bq_spearman / teacher_spearman * 100,
            'storage_bytes': storage_bytes,
            'bits_per_dim': 1,
        }
        
        results['methods'].append(method_result)
        print(f"  BQ-{bq_dim}: ρ = {bq_spearman:.4f} ({method_result['retention']:.1f}% retention, {storage_bytes} bytes)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADD CGT RESULT
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGT uses lorentz_similarity from HARDENED module
    cgt_storage = (config.cgt_dim + 1) * 4  # float32, +1 for time component
    
    results['methods'].append({
        'method': f'CGT-{config.cgt_dim}',
        'dim': config.cgt_dim + 1,  # ambient dim
        'spearman': cgt_spearman,
        'retention': cgt_spearman / teacher_spearman * 100,
        'storage_bytes': cgt_storage,
        'bits_per_dim': 32,
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("COMPARISON: CGT vs BQ")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'Spearman':>10} {'Retention':>12} {'Storage':>12}")
    print("-" * 55)
    
    print(f"{'Teacher':<15} {teacher_spearman:>10.4f} {'100.0%':>12} {teacher_dim*4:>10} B")
    
    for method in results['methods']:
        ret_str = f"{method['retention']:.1f}%"
        print(f"{method['method']:<15} {method['spearman']:>10.4f} {ret_str:>12} {method['storage_bytes']:>10} B")
    
    # Find best BQ that beats CGT in storage
    bq_methods = [m for m in results['methods'] if m['method'].startswith('BQ')]
    bq_better_storage = [m for m in bq_methods if m['storage_bytes'] < cgt_storage]
    
    if bq_better_storage:
        best_bq = max(bq_better_storage, key=lambda x: x['spearman'])
        cgt_method = next(m for m in results['methods'] if m['method'].startswith('CGT'))
        
        print(f"\n📊 Best BQ with better storage than CGT: {best_bq['method']}")
        print(f"   BQ:  ρ = {best_bq['spearman']:.4f}, storage = {best_bq['storage_bytes']} bytes")
        print(f"   CGT: ρ = {cgt_method['spearman']:.4f}, storage = {cgt_method['storage_bytes']} bytes")
        print(f"   Quality difference: {cgt_method['spearman'] - best_bq['spearman']:+.4f}")
        
        results['best_bq_comparison'] = {
            'bq_method': best_bq['method'],
            'bq_spearman': best_bq['spearman'],
            'bq_storage': best_bq['storage_bytes'],
            'cgt_spearman': cgt_method['spearman'],
            'cgt_storage': cgt_method['storage_bytes'],
            'quality_advantage': cgt_method['spearman'] - best_bq['spearman'],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'bq_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    pd.DataFrame(results['methods']).to_csv(
        output_dir / 'bq_comparison_results.csv', index=False
    )
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.3 - BINARY QUANTIZATION COMPARISON")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed embeddings.")
    print("   Use run_bq_comparison() with your data.")
