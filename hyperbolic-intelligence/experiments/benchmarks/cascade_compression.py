# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part I.19 - Cascade Compression Analysis
========================================

AUDIT COMPLIANCE:
- NO geometry derivation
- Quantization is pure arithmetic (not geometry)
- Uses hardened modules for CGT evaluation

PURPOSE:
Measure post-CGT compressibility via cascade quantization:
1. Scalar Quantization (Int8) - 4× compression
2. Product Quantization (4-bit effective) - 8× compression  
3. Binary Quantization (1-bit) - 32× compression

OPERATION:
Quantization acts AFTER CGT embeddings are computed.
No retraining, no geometry change.

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
#                    QUANTIZER CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ScalarQuantizer:
    """
    Scalar Quantization: Float32 → Int8.
    
    Linear mapping of min/max to [-128, 127].
    Compression: 4× (32 bits → 8 bits)
    
    OPERATION:
    - Purely arithmetic (no geometry)
    - Quantize each dimension independently
    """
    
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.scale = None
    
    def fit(self, X: np.ndarray) -> "ScalarQuantizer":
        """
        Fit quantizer to data range.
        
        Args:
            X: Embeddings [N, D]
        
        Returns:
            self
        """
        self.min_val = X.min()
        self.max_val = X.max()
        self.scale = 255 / (self.max_val - self.min_val + 1e-9)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize to Int8 and dequantize for similarity calculation.
        
        Args:
            X: Embeddings [N, D]
        
        Returns:
            Dequantized embeddings [N, D]
        """
        # Quantize
        X_int = np.round((X - self.min_val) * self.scale - 128).astype(np.int8)
        
        # Dequantize (for similarity calculation)
        X_deq = (X_int.astype(np.float32) + 128) / self.scale + self.min_val
        
        return X_deq
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs Float32."""
        return 4.0  # 32 bits → 8 bits


class BinaryQuantizer:
    """
    Binary Quantization: Float32 → Binary (sign bit).
    
    Compression: 32× (32 bits → 1 bit)
    
    OPERATION:
    - Purely arithmetic (sign function)
    - Each value becomes ±1
    """
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
    
    def fit(self, X: np.ndarray) -> "BinaryQuantizer":
        """Fit is a no-op for binary quantization."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize to binary.
        
        Args:
            X: Embeddings [N, D]
        
        Returns:
            Binary embeddings (-1 or +1) [N, D]
        """
        return np.sign(X - self.threshold)
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs Float32."""
        return 32.0  # 32 bits → 1 bit


class ProductQuantizer:
    """
    Product Quantization: Float32 → PQ codes.
    
    Divides vector into subvectors and quantizes each with K-means.
    
    OPERATION:
    - Uses sklearn KMeans (standard library)
    - No geometry derivation
    
    Parameters:
        n_subvectors: Number of subvectors (M)
        n_clusters: Clusters per subvector (K=256 for 8-bit codes)
    
    Effective bits: M * log2(K) = 4 * 8 = 32 bits total
    With default settings: ~4× compression (similar to scalar)
    With K=16: 4 * 4 = 16 bits → 8× compression
    """
    
    def __init__(self, n_subvectors: int = 4, n_clusters: int = 256):
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.codebooks: List[KMeans] = []
        self.subvector_size: Optional[int] = None
    
    def fit(self, X: np.ndarray) -> "ProductQuantizer":
        """
        Fit PQ codebooks.
        
        Args:
            X: Embeddings [N, D]
        
        Returns:
            self
        """
        n_samples, dim = X.shape
        self.subvector_size = dim // self.n_subvectors
        
        self.codebooks = []
        for m in range(self.n_subvectors):
            start = m * self.subvector_size
            end = start + self.subvector_size
            
            subvector = X[:, start:end]
            
            # Fit KMeans (standard sklearn)
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=1,
                max_iter=50,
                random_state=42,
            )
            kmeans.fit(subvector)
            self.codebooks.append(kmeans)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode and decode through PQ codebooks.
        
        Args:
            X: Embeddings [N, D]
        
        Returns:
            Reconstructed embeddings [N, D]
        """
        n_samples = X.shape[0]
        reconstructed = np.zeros_like(X)
        
        for m, kmeans in enumerate(self.codebooks):
            start = m * self.subvector_size
            end = start + self.subvector_size
            
            subvector = X[:, start:end]
            
            # Encode (find nearest centroid)
            codes = kmeans.predict(subvector)
            
            # Decode (replace with centroid)
            reconstructed[:, start:end] = kmeans.cluster_centers_[codes]
        
        return reconstructed
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs Float32."""
        # Each subvector needs log2(n_clusters) bits
        bits_per_subvector = np.log2(self.n_clusters)
        total_bits = self.n_subvectors * bits_per_subvector
        return 32.0 * 32 / total_bits  # Assuming 32D input


# ═══════════════════════════════════════════════════════════════════════════════
#                    CASCADE COMPRESSION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_cascade_compression(
    cgt_emb1: np.ndarray,
    cgt_emb2: np.ndarray,
    scores: np.ndarray,
    original_spearman: float,
) -> Dict[str, Dict]:
    """
    Evaluate cascade compression methods.
    
    Args:
        cgt_emb1: CGT embeddings for first sentences [N, D]
        cgt_emb2: CGT embeddings for second sentences [N, D]
        scores: Ground truth similarity scores [N]
        original_spearman: CGT baseline Spearman (before quantization)
    
    Returns:
        Dictionary with results for each quantization method
    """
    # Combine embeddings for fitting
    all_emb = np.concatenate([cgt_emb1, cgt_emb2], axis=0)
    
    results = {}
    
    # 1. Scalar Quantization (Int8)
    print("  Evaluating Scalar Quantization (Int8)...")
    sq = ScalarQuantizer().fit(all_emb)
    sq_emb1 = sq.transform(cgt_emb1)
    sq_emb2 = sq.transform(cgt_emb2)
    
    # Cosine similarity (standard)
    sq_sims = np.sum(sq_emb1 * sq_emb2, axis=-1) / (
        np.linalg.norm(sq_emb1, axis=-1) * np.linalg.norm(sq_emb2, axis=-1) + 1e-8
    )
    sq_spearman, _ = spearmanr(sq_sims, scores)
    
    results['scalar_int8'] = {
        'spearman': float(sq_spearman),
        'retention_vs_cgt': float(sq_spearman / original_spearman * 100),
        'compression_ratio': sq.compression_ratio,
        'bits_per_dim': 8,
    }
    
    # 2. Product Quantization
    print("  Evaluating Product Quantization (4-bit)...")
    pq = ProductQuantizer(n_subvectors=8, n_clusters=16).fit(all_emb)  # 4 bits per subvector
    pq_emb1 = pq.transform(cgt_emb1)
    pq_emb2 = pq.transform(cgt_emb2)
    
    pq_sims = np.sum(pq_emb1 * pq_emb2, axis=-1) / (
        np.linalg.norm(pq_emb1, axis=-1) * np.linalg.norm(pq_emb2, axis=-1) + 1e-8
    )
    pq_spearman, _ = spearmanr(pq_sims, scores)
    
    results['product_4bit'] = {
        'spearman': float(pq_spearman),
        'retention_vs_cgt': float(pq_spearman / original_spearman * 100),
        'compression_ratio': 8.0,  # 32 bits → 4 bits effective
        'bits_per_dim': 4,
    }
    
    # 3. Binary Quantization
    print("  Evaluating Binary Quantization (1-bit)...")
    bq = BinaryQuantizer().fit(all_emb)
    bq_emb1 = bq.transform(cgt_emb1)
    bq_emb2 = bq.transform(cgt_emb2)
    
    # Hamming similarity for binary: (D - hamming_distance) / D
    # Equivalently: inner product of ±1 vectors / D
    bq_sims = np.sum(bq_emb1 * bq_emb2, axis=-1) / bq_emb1.shape[-1]
    bq_spearman, _ = spearmanr(bq_sims, scores)
    
    results['binary_1bit'] = {
        'spearman': float(bq_spearman),
        'retention_vs_cgt': float(bq_spearman / original_spearman * 100),
        'compression_ratio': bq.compression_ratio,
        'bits_per_dim': 1,
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    CASCADE COMPRESSION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cascade_compression(
    cgt_emb1: torch.Tensor,
    cgt_emb2: torch.Tensor,
    test_scores: torch.Tensor,
    cgt_spearman: float,
    teacher_spearman: float,
    output_dir: Path,
) -> Dict:
    """
    Run cascade compression analysis.
    
    Args:
        cgt_emb1: CGT embeddings for first sentences [N, D+1]
        cgt_emb2: CGT embeddings for second sentences [N, D+1]
        test_scores: Ground truth similarity scores [N]
        cgt_spearman: CGT baseline Spearman
        teacher_spearman: Teacher baseline Spearman
        output_dir: Output directory for results
    
    Returns:
        Dictionary with cascade compression results
    """
    print("\n" + "=" * 70)
    print("PART I.19 - CASCADE COMPRESSION ANALYSIS")
    print("=" * 70)
    
    # Convert to numpy and extract spatial components
    # CGT embeddings have shape [N, D+1] where D+1 includes time component
    # For quantization, we use spatial components only (indices 1:)
    cgt_emb1_np = cgt_emb1[:, 1:].cpu().numpy()  # Spatial components
    cgt_emb2_np = cgt_emb2[:, 1:].cpu().numpy()
    scores_np = test_scores.cpu().numpy()
    
    print(f"\nCGT baseline: ρ = {cgt_spearman:.4f} ({cgt_spearman/teacher_spearman*100:.1f}% of Teacher)")
    print(f"Embedding dimension (spatial): {cgt_emb1_np.shape[1]}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'cgt_baseline_spearman': cgt_spearman,
        'teacher_spearman': teacher_spearman,
        'spatial_dim': cgt_emb1_np.shape[1],
    }
    
    # Evaluate cascade compression
    print("\nEvaluating cascade compression methods...")
    cascade_results = evaluate_cascade_compression(
        cgt_emb1_np, cgt_emb2_np, scores_np, cgt_spearman
    )
    
    results['cascade'] = cascade_results
    
    # Print summary
    print("\n" + "-" * 70)
    print("CASCADE COMPRESSION RESULTS")
    print("-" * 70)
    print(f"{'Method':<20} {'Spearman':>10} {'vs CGT':>10} {'vs Teacher':>12} {'Bits/Dim':>10}")
    print("-" * 70)
    
    # Baseline CGT
    print(f"{'CGT (Float32)':<20} {cgt_spearman:>10.4f} {'100.0%':>10} {cgt_spearman/teacher_spearman*100:>11.1f}% {'32':>10}")
    
    # Quantized versions
    for method, data in cascade_results.items():
        name = {
            'scalar_int8': 'CGT + Int8',
            'product_4bit': 'CGT + PQ-4bit',
            'binary_1bit': 'CGT + Binary',
        }[method]
        
        print(f"{name:<20} {data['spearman']:>10.4f} {data['retention_vs_cgt']:>9.1f}% "
              f"{data['spearman']/teacher_spearman*100:>11.1f}% {data['bits_per_dim']:>10}")
    
    # Storage calculation
    print("\n" + "-" * 70)
    print("STORAGE COMPARISON (per embedding)")
    print("-" * 70)
    
    dim = cgt_emb1_np.shape[1]
    storage = {
        'Teacher (384D)': 384 * 4,
        'CGT (Float32)': (dim + 1) * 4,  # +1 for time component
        'CGT + Int8': (dim + 1) * 1,
        'CGT + PQ-4bit': (dim + 1) * 0.5,
        'CGT + Binary': (dim + 1) / 8,
    }
    
    for method, bytes_val in storage.items():
        print(f"{method:<20}: {bytes_val:>8.1f} bytes")
    
    results['storage_bytes'] = storage
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    with open(output_dir / 'cascade_compression_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV summary
    rows = [
        {'method': 'Teacher', 'spearman': teacher_spearman, 'retention_teacher': 100.0,
         'bits_per_dim': 32, 'storage_bytes': 384 * 4},
        {'method': 'CGT (Float32)', 'spearman': cgt_spearman, 
         'retention_teacher': cgt_spearman/teacher_spearman*100,
         'bits_per_dim': 32, 'storage_bytes': (dim + 1) * 4},
    ]
    for method, data in cascade_results.items():
        rows.append({
            'method': f'CGT + {method}',
            'spearman': data['spearman'],
            'retention_teacher': data['spearman']/teacher_spearman*100,
            'bits_per_dim': data['bits_per_dim'],
            'storage_bytes': (dim + 1) * data['bits_per_dim'] / 8,
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / 'cascade_compression_summary.csv', index=False)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART I.19 - CASCADE COMPRESSION ANALYSIS")
    print("=" * 80)
    
    print("\n⚠️ This module requires pre-computed CGT embeddings.")
    print("   Use run_cascade_compression() with your data.")
