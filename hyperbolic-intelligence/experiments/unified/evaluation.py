# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Unified Evaluation Pipeline
===========================

SINGLE evaluation pipeline for ALL models.
Based EXCLUSIVELY on k_Lighting_NUMERICAL_PARITY.ipynb methodology.

AUDIT COMPLIANCE:
- ✅ Same tests for all models
- ✅ Same metrics for all models
- ✅ Same order for all models
- ✅ No new methodological decisions
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cgt.utils.helpers import set_global_seed, get_device
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig


def enforce_float64():
    """Enforce float64 precision."""
    torch.set_default_dtype(torch.float64)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False


# ═══════════════════════════════════════════════════════════════════════════════
#                    EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb

@dataclass
class EvaluationResult:
    """Results from evaluation pipeline."""
    model_name: str
    
    # STS-B Performance
    test_spearman: float
    test_pearson: float
    val_spearman: float
    
    # Teacher Comparison
    teacher_spearman: float
    retention_percent: float
    
    # Compression Stats
    teacher_dim: int
    student_dim: int
    compression_ratio: float
    
    # Storage
    model_size_bytes: int
    embedding_size_bytes: int
    
    # Falsification Tests
    f1_projection_passed: bool
    f2_distance_passed: bool
    f3_topological_passed: bool
    
    # Metadata
    timestamp: str
    device: str
    dtype: str


def compute_spearman(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    substrate: LorentzSubstrateHardened,
) -> float:
    """
    Compute Spearman correlation using Lorentz distances.
    
    SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    """
    with torch.no_grad():
        dists = substrate.dist(emb1, emb2)
        sims = -dists  # Negative distance = similarity
        rho, _ = spearmanr(sims.cpu().numpy(), scores.cpu().numpy())
    return rho


def compute_pearson(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    scores: torch.Tensor,
    substrate: LorentzSubstrateHardened,
) -> float:
    """
    Compute Pearson correlation using Lorentz distances.
    
    SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    """
    with torch.no_grad():
        dists = substrate.dist(emb1, emb2)
        sims = -dists
        r, _ = pearsonr(sims.cpu().numpy(), scores.cpu().numpy())
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#                    FALSIFICATION TESTS (F1-F3)
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb

def f1_projection_integrity(
    embeddings: torch.Tensor,
    substrate: LorentzSubstrateHardened,
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """
    F1: Verify embeddings lie on the hyperboloid.
    
    Checks: x₀² - x₁² - ... - xₙ² = -1/c
    
    SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    """
    with torch.no_grad():
        # Minkowski inner product: x₀² - sum(xᵢ²)
        time_comp = embeddings[:, 0:1]
        space_comp = embeddings[:, 1:]
        
        inner = time_comp**2 - (space_comp**2).sum(dim=1, keepdim=True)
        
        # Should equal -1/c (for c=-1, this is 1)
        target = -1.0 / substrate.curvature
        error = torch.abs(inner - target).mean().item()
        
        passed = error < tolerance
    
    return passed, error


def f2_distance_preservation(
    student_emb1: torch.Tensor,
    student_emb2: torch.Tensor,
    teacher_emb1: torch.Tensor,
    teacher_emb2: torch.Tensor,
    substrate: LorentzSubstrateHardened,
    threshold: float = 0.7,
) -> Tuple[bool, float]:
    """
    F2: Verify distance correlation with teacher.
    
    SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    """
    with torch.no_grad():
        # Student distances (Lorentz)
        student_dists = substrate.dist(student_emb1, student_emb2)
        
        # Teacher distances (Euclidean/cosine)
        teacher_sims = torch.nn.functional.cosine_similarity(teacher_emb1, teacher_emb2)
        teacher_dists = 1 - teacher_sims  # Convert to distance
        
        # Correlation
        rho, _ = spearmanr(
            student_dists.cpu().numpy(),
            teacher_dists.cpu().numpy()
        )
        
        passed = rho > threshold
    
    return passed, rho


def f3_topological_consistency(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    substrate: LorentzSubstrateHardened,
    k: int = 10,
    threshold: float = 0.8,
) -> Tuple[bool, float]:
    """
    F3: Verify k-NN neighborhood preservation.
    
    SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
    
    AUDIT FIX (2026-01-19):
    - Changed from Euclidean metric to Lorentz geodesic distance
    - Euclidean distance on hyperboloid coordinates is INVALID
    - Must use intrinsic geodesic distance for hyperbolic space
    - See: Gemini DeepSearch Falsification Audit Report
    """
    n_samples = min(1000, student_embeddings.shape[0])  # Limit for efficiency
    
    # Sample indices
    indices = torch.randperm(student_embeddings.shape[0])[:n_samples]
    
    student_sample = student_embeddings[indices]
    teacher_sample = teacher_embeddings[indices].cpu().numpy()
    
    # Compute pairwise distances
    # AUDIT FIX: Use Lorentz geodesic distance for student (hyperbolic space)
    # NOT Euclidean chordal distance which distorts neighborhood structure
    with torch.no_grad():
        # Compute full distance matrix using substrate.dist
        student_dists_list = []
        for i in range(n_samples):
            # Distance from point i to all other points
            point_i = student_sample[i:i+1].expand(n_samples, -1)
            dists_i = substrate.dist(point_i, student_sample)
            student_dists_list.append(dists_i.cpu().numpy())
        student_dists = np.stack(student_dists_list, axis=0)
    
    # Teacher distances (Euclidean/cosine) - unchanged
    teacher_dists = cdist(teacher_sample, teacher_sample, metric='cosine')
    
    # Get k-NN for each sample
    overlaps = []
    for i in range(n_samples):
        student_knn = set(np.argsort(student_dists[i])[:k+1]) - {i}
        teacher_knn = set(np.argsort(teacher_dists[i])[:k+1]) - {i}
        
        overlap = len(student_knn & teacher_knn) / k
        overlaps.append(overlap)
    
    mean_overlap = np.mean(overlaps)
    passed = mean_overlap > threshold
    
    return passed, mean_overlap


# ═══════════════════════════════════════════════════════════════════════════════
#                    STORAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb

def compute_model_size(checkpoint_path: Path) -> int:
    """Compute model size in bytes."""
    if checkpoint_path.exists():
        return checkpoint_path.stat().st_size
    return 0


def compute_embedding_size(
    n_samples: int,
    dim: int,
    dtype_bytes: int = 8,  # float64
) -> int:
    """Compute embedding storage size."""
    return n_samples * dim * dtype_bytes


# ═══════════════════════════════════════════════════════════════════════════════
#                    UNIFIED EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedEvaluator:
    """
    Single evaluator for all models.
    
    Uses EXACTLY the pipeline from k_Lighting_NUMERICAL_PARITY.ipynb.
    No methodological variations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        enforce_float64()
        self.device = device if device else get_device()
        self.dtype = torch.float64
        
        # Results storage
        self.results: Dict[str, EvaluationResult] = {}
    
    def evaluate_model(
        self,
        model_name: str,
        model: torch.nn.Module,
        substrate: LorentzSubstrateHardened,
        test_emb1: torch.Tensor,
        test_emb2: torch.Tensor,
        test_scores: torch.Tensor,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
        teacher_emb1: torch.Tensor,
        teacher_emb2: torch.Tensor,
        teacher_spearman: float,
        teacher_dim: int,
        checkpoint_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single model using the unified pipeline.
        
        SOURCE: k_Lighting_NUMERICAL_PARITY.ipynb
        """
        model.eval()
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Ensure float64
        test_emb1 = test_emb1.to(self.dtype).to(self.device)
        test_emb2 = test_emb2.to(self.dtype).to(self.device)
        test_scores = test_scores.to(self.dtype)
        val_emb1 = val_emb1.to(self.dtype).to(self.device)
        val_emb2 = val_emb2.to(self.dtype).to(self.device)
        val_scores = val_scores.to(self.dtype)
        teacher_emb1 = teacher_emb1.to(self.dtype).to(self.device)
        teacher_emb2 = teacher_emb2.to(self.dtype).to(self.device)
        
        # Get student embeddings
        with torch.no_grad():
            student_test_emb1 = model(test_emb1)
            student_test_emb2 = model(test_emb2)
            student_val_emb1 = model(val_emb1)
            student_val_emb2 = model(val_emb2)
        
        student_dim = student_test_emb1.shape[1]
        
        # === STS-B METRICS ===
        print("\n[1/4] Computing STS-B metrics...")
        
        test_spearman = compute_spearman(
            student_test_emb1, student_test_emb2, test_scores, substrate
        )
        test_pearson = compute_pearson(
            student_test_emb1, student_test_emb2, test_scores, substrate
        )
        val_spearman = compute_spearman(
            student_val_emb1, student_val_emb2, val_scores, substrate
        )
        
        retention = test_spearman / teacher_spearman * 100
        
        print(f"  Test Spearman: {test_spearman:.4f}")
        print(f"  Test Pearson: {test_pearson:.4f}")
        print(f"  Val Spearman: {val_spearman:.4f}")
        print(f"  Retention: {retention:.1f}%")
        
        # === FALSIFICATION TESTS ===
        print("\n[2/4] Running falsification tests...")
        
        # Combine test embeddings for F1
        all_student_emb = torch.cat([student_test_emb1, student_test_emb2], dim=0)
        all_teacher_emb = torch.cat([teacher_emb1, teacher_emb2], dim=0)
        
        f1_passed, f1_error = f1_projection_integrity(all_student_emb, substrate)
        print(f"  F1 (Projection): {'PASS' if f1_passed else 'FAIL'} (error={f1_error:.2e})")
        
        f2_passed, f2_corr = f2_distance_preservation(
            student_test_emb1, student_test_emb2,
            teacher_emb1, teacher_emb2,
            substrate
        )
        print(f"  F2 (Distance): {'PASS' if f2_passed else 'FAIL'} (corr={f2_corr:.4f})")
        
        f3_passed, f3_overlap = f3_topological_consistency(
            all_student_emb, all_teacher_emb, substrate
        )
        print(f"  F3 (Topological): {'PASS' if f3_passed else 'FAIL'} (overlap={f3_overlap:.4f})")
        
        # === STORAGE ANALYSIS ===
        print("\n[3/4] Computing storage metrics...")
        
        model_size = compute_model_size(checkpoint_path) if checkpoint_path else 0
        embedding_size = compute_embedding_size(
            n_samples=test_emb1.shape[0] * 2,  # Both sentence pairs
            dim=student_dim,
        )
        compression_ratio = teacher_dim / student_dim
        
        print(f"  Model size: {model_size / 1024:.1f} KB")
        print(f"  Embedding size: {embedding_size / 1024:.1f} KB")
        print(f"  Compression: {compression_ratio:.1f}x ({teacher_dim}d → {student_dim}d)")
        
        # === COMPILE RESULTS ===
        print("\n[4/4] Compiling results...")
        
        result = EvaluationResult(
            model_name=model_name,
            test_spearman=test_spearman,
            test_pearson=test_pearson,
            val_spearman=val_spearman,
            teacher_spearman=teacher_spearman,
            retention_percent=retention,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            compression_ratio=compression_ratio,
            model_size_bytes=model_size,
            embedding_size_bytes=embedding_size,
            f1_projection_passed=f1_passed,
            f2_distance_passed=f2_passed,
            f3_topological_passed=f3_passed,
            timestamp=datetime.now().isoformat(),
            device=str(self.device),
            dtype=str(self.dtype),
        )
        
        self.results[model_name] = result
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {model_name}")
        print(f"  ρ = {test_spearman:.4f} | Retention = {retention:.1f}%")
        print(f"  Falsification: F1={'✓' if f1_passed else '✗'} F2={'✓' if f2_passed else '✗'} F3={'✓' if f3_passed else '✗'}")
        print(f"{'='*60}")
        
        return result
    
    def get_results_table(self) -> str:
        """
        Generate results table in specified format.
        
        FORMAT (from specification):
        Modelo | Teacher | Dim orig | Dim comp | ρ | Retention | Storage | Falsification | Obs
        """
        if not self.results:
            return "No results available."
        
        # Sort by test_spearman (descending) for ranking
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.test_spearman,
            reverse=True
        )
        
        # Header
        header = (
            f"{'Modelo':<30} | {'Teacher':<25} | {'Dim Orig':>8} | {'Dim Comp':>8} | "
            f"{'ρ (Spearman)':>12} | {'Retention':>10} | {'Storage':>12} | "
            f"{'Falsif':>8} | {'Obs':<15}"
        )
        separator = "-" * len(header)
        
        lines = [separator, header, separator]
        
        for rank, result in enumerate(sorted_results, 1):
            # Determine teacher name
            if result.teacher_dim == 768:
                teacher = "all-mpnet-base-v2"
            else:
                teacher = "all-MiniLM-L6-v2"
            
            # Falsification summary
            f_tests = f"{'✓' if result.f1_projection_passed else '✗'}" \
                      f"{'✓' if result.f2_distance_passed else '✗'}" \
                      f"{'✓' if result.f3_topological_passed else '✗'}"
            
            # Observations
            obs = f"Rank #{rank}"
            if result.retention_percent > 90:
                obs += " ★"
            
            line = (
                f"{result.model_name:<30} | {teacher:<25} | "
                f"{result.teacher_dim:>8} | {result.student_dim:>8} | "
                f"{result.test_spearman:>12.4f} | {result.retention_percent:>9.1f}% | "
                f"{result.model_size_bytes/1024:>10.1f}KB | "
                f"{f_tests:>8} | {obs:<15}"
            )
            lines.append(line)
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: Path) -> None:
        """Save all results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON results
        results_dict = {
            name: asdict(result)
            for name, result in self.results.items()
        }
        
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Table
        table_path = output_dir / "results_table.txt"
        with open(table_path, "w") as f:
            f.write(self.get_results_table())
        
        print(f"\nResults saved to:")
        print(f"  {json_path}")
        print(f"  {table_path}")
