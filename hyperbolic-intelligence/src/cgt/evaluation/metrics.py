# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Evaluation Module
=====================

Scientific validation and evaluation metrics for CGT.

This module implements:
- Falsification protocols (F1-F3) for geometric property validation
- STS-B evaluation with metric-consistent similarity
- Geometric health metrics (ERank, Gromov δ, distortion)

Falsification Protocols
-----------------------
F1 (Homotopy): β₀ stability under perturbation
F2 (Stability): Bounded perturbation amplification
F3 (Forman-Ricci): Geometric curvature consistency

Mathematical Status
-------------------
- F1: Empirical homotopy test (not formal proof)
- F2: Local Lipschitz estimation (not global bound)
- F3: Discrete Forman-Ricci (proxy for continuous curvature)

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress, spearmanr

from cgt.geometry import LorentzSubstrate
from cgt.losses import TopoLoss


class FalsificationProtocols:
    """
    Scientific falsification protocols F1-F3.

    Implements stress tests to validate or falsify claims about
    the geometric properties of the CGT embedding.

    Protocols:
    - F1 (Homotopy): Tests β₀ stability under input perturbation
    - F2 (Stability): Measures perturbation amplification
    - F3 (Forman-Ricci): Checks geometric curvature consistency

    Attributes:
        model: CGT student model.
        substrate: Lorentz substrate.
        topo: Topological loss for β₀ computation.
        results: Dictionary storing test results.

    Notes:
        - Space: Tests operate on manifold H^n
        - Status: Empirical falsification (not formal proofs)
        - These tests can DISPROVE but not PROVE properties
    """

    def __init__(
        self,
        model: nn.Module,
        substrate: LorentzSubstrate,
        target_beta_0: float = 1.0,
    ):
        """
        Initialize falsification protocols.

        Args:
            model: CGT student model.
            substrate: Lorentz substrate.
            target_beta_0: Target connectivity for F1.
        """
        self.model = model
        self.substrate = substrate
        self.topo = TopoLoss(target_beta_0)
        self.results: Dict[str, Optional[Dict]] = {
            "F1": None,
            "F2": None,
            "F3": None,
        }

    @torch.no_grad()
    def F1_homotopy(
        self,
        teacher_emb: torch.Tensor,
        hyp_emb: torch.Tensor,
        perturbation_sigma: float = 0.1,
    ) -> Dict:
        """
        F1: Test homotopy preservation.

        Verifies that β₀ (connectivity) is stable under small
        perturbations to the input embeddings.

        Args:
            teacher_emb: Teacher embeddings [B, D].
            hyp_emb: Current hyperbolic embeddings [B, n+1].
            perturbation_sigma: Noise magnitude.

        Returns:
            Dictionary with original/perturbed β₀ and pass status.

        Notes:
            - Space: Topological comparison on H^n
            - Status: Empirical test (single perturbation sample)
            - Threshold: δ(β₀) < 0.5 indicates stability
            - Does NOT guarantee homotopy equivalence
        """
        # Original β₀
        D_orig = self.substrate.distance_matrix(hyp_emb)
        _, beta_0_orig = self.topo(D_orig)

        # Perturbed β₀
        noise = torch.randn_like(teacher_emb) * perturbation_sigma
        hyp_pert = self.model(teacher_emb + noise, use_homeostatic=False)
        D_pert = self.substrate.distance_matrix(hyp_pert)
        _, beta_0_pert = self.topo(D_pert)

        delta = abs(beta_0_orig - beta_0_pert)
        passed = delta < 0.5

        result = {
            "beta_0_original": beta_0_orig,
            "beta_0_perturbed": beta_0_pert,
            "delta": delta,
            "threshold": 0.5,
            "passed": passed,
        }
        self.results["F1"] = result
        return result

    @torch.no_grad()
    def F2_encoder_stability(
        self,
        teacher_emb: torch.Tensor,
        hyp_emb: torch.Tensor,
        perturbation_sigma: float = 0.1,
        n_steps: int = 20,
        max_amplification: float = 5.0,
    ) -> Dict:
        """
        F2: Test encoder stability (Lipschitz-like property).

        Measures how the encoder amplifies input perturbations.
        A stable encoder should have bounded amplification.

        Args:
            teacher_emb: Teacher embeddings [B, D].
            hyp_emb: Current hyperbolic embeddings [B, n+1].
            perturbation_sigma: Maximum noise magnitude.
            n_steps: Number of noise levels to test.
            max_amplification: Threshold for stability.

        Returns:
            Dictionary with amplification slope and pass status.

        Notes:
            - Space: Input-output distance ratio analysis
            - Status: First-order local estimate
            - Tests at multiple noise levels via linear regression
            - Does NOT provide global Lipschitz bound
        """
        sigmas = np.linspace(0.01, perturbation_sigma, n_steps)
        d_inputs, d_outputs = [], []

        for sigma in sigmas:
            noise = torch.randn_like(teacher_emb) * sigma
            hyp_pert = self.model(teacher_emb + noise, use_homeostatic=False)

            d_in = noise.norm(dim=-1).mean().item()
            d_out = self.substrate.dist(hyp_emb, hyp_pert).mean().item()

            d_inputs.append(d_in)
            d_outputs.append(d_out)

        # Linear regression for slope (amplification factor)
        d_inputs = np.array(d_inputs, dtype=np.float64)
        d_outputs = np.array(d_outputs, dtype=np.float64)

        try:
            slope, intercept, r_value, _, _ = linregress(d_inputs, d_outputs)
            passed = slope < max_amplification
        except Exception:
            slope, intercept, r_value = float("nan"), 0.0, 0.0
            passed = False

        result = {
            "amplification_slope": slope,
            "r_squared": r_value**2 if not np.isnan(r_value) else 0.0,
            "threshold": max_amplification,
            "passed": passed,
            "d_inputs": d_inputs.tolist(),
            "d_outputs": d_outputs.tolist(),
        }
        self.results["F2"] = result
        return result

    @torch.no_grad()
    def F3_forman_ricci(
        self,
        hyp_emb: torch.Tensor,
        knn_neighbors: int = 10,
        margin: int = 10,
    ) -> Dict:
        """
        F3: Test geometric consistency via Forman-Ricci curvature.

        Computes discrete Forman-Ricci curvature on the k-NN graph
        to verify geometric structure preservation.

        Args:
            hyp_emb: Hyperbolic embeddings [B, n+1].
            knn_neighbors: Number of neighbors for graph.
            margin: Tolerance for curvature range.

        Returns:
            Dictionary with curvature statistics and pass status.

        Notes:
            - Space: k-NN graph on H^n
            - Status: Discrete proxy for continuous curvature
            - Forman-Ricci measures local graph structure
            - Does NOT correspond to manifold sectional curvature
        """
        B = hyp_emb.shape[0]
        k = min(knn_neighbors, B - 1)

        # Distance matrix
        D = self.substrate.distance_matrix(hyp_emb)

        # k-NN adjacency
        _, indices = torch.topk(-D, k + 1, dim=1)
        indices = indices[:, 1:]  # Exclude self

        # Build adjacency matrix
        adj = torch.zeros(B, B, device=hyp_emb.device)
        for i in range(B):
            adj[i, indices[i]] = 1.0
            adj[indices[i], i] = 1.0  # Symmetrize

        # Compute degrees
        degrees = adj.sum(dim=1)

        # Forman-Ricci curvature per edge
        # κ(e) = 4 - deg(u) - deg(v) for edge (u,v)
        curvatures = []
        for i in range(B):
            for j in range(i + 1, B):
                if adj[i, j] > 0:
                    kappa = 4.0 - degrees[i].item() - degrees[j].item()
                    curvatures.append(kappa)

        if len(curvatures) == 0:
            curvatures = [0.0]

        curvatures = np.array(curvatures)
        mean_curv = curvatures.mean()
        std_curv = curvatures.std()

        # Pass if curvature is in reasonable range
        passed = abs(mean_curv) < margin

        result = {
            "mean_curvature": mean_curv,
            "std_curvature": std_curv,
            "min_curvature": curvatures.min(),
            "max_curvature": curvatures.max(),
            "threshold": margin,
            "passed": passed,
        }
        self.results["F3"] = result
        return result

    def run_all(
        self,
        teacher_emb: torch.Tensor,
        hyp_emb: torch.Tensor,
        perturbation_sigma: float = 0.1,
        enable_f1: bool = True,
        enable_f2: bool = True,
        enable_f3: bool = True,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run all enabled falsification protocols.

        Args:
            teacher_emb: Teacher embeddings.
            hyp_emb: Hyperbolic embeddings.
            perturbation_sigma: Noise magnitude for F1/F2.
            enable_f1: Run F1 homotopy test.
            enable_f2: Run F2 stability test.
            enable_f3: Run F3 Forman-Ricci test.

        Returns:
            Dictionary with all test results.
        """
        if enable_f1:
            self.F1_homotopy(teacher_emb, hyp_emb, perturbation_sigma)
        if enable_f2:
            self.F2_encoder_stability(teacher_emb, hyp_emb, perturbation_sigma)
        if enable_f3:
            self.F3_forman_ricci(hyp_emb)

        return self.results


@torch.no_grad()
def evaluate_stsb(
    model: nn.Module,
    sent1_list: List[str],
    sent2_list: List[str],
    scores: torch.Tensor,
    sent_to_idx: Dict[str, int],
    teacher_emb: torch.Tensor,
    use_geodesic: bool = True,
    device: torch.device = None,
) -> float:
    """
    Evaluate on STS-B benchmark.

    Args:
        model: CGT student model.
        sent1_list: First sentences.
        sent2_list: Second sentences.
        scores: Ground truth similarity scores.
        sent_to_idx: Sentence to index mapping.
        teacher_emb: Pre-computed teacher embeddings.
        use_geodesic: Use geodesic similarity (recommended for consistency).
        device: Computation device.

    Returns:
        Spearman correlation coefficient.

    Notes:
        - Space: Similarity on H^n
        - Status: Exact evaluation metric
        - use_geodesic=True ensures train/eval consistency
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Project all embeddings
    hyp_emb = model(teacher_emb.to(device), use_homeostatic=False)

    preds = []
    for s1, s2 in zip(sent1_list, sent2_list):
        idx1 = sent_to_idx.get(s1, 0)
        idx2 = sent_to_idx.get(s2, 0)

        h1 = hyp_emb[idx1 : idx1 + 1]
        h2 = hyp_emb[idx2 : idx2 + 1]

        if use_geodesic:
            sim = model.substrate.geodesic_similarity(h1, h2).item()
        else:
            # Cosine similarity on spatial coordinates
            h1_spatial = F.normalize(h1[:, 1:], dim=-1)
            h2_spatial = F.normalize(h2[:, 1:], dim=-1)
            sim = F.cosine_similarity(h1_spatial, h2_spatial).item()

        preds.append(sim)

    rho, _ = spearmanr(preds, scores.cpu().numpy())
    return rho if not np.isnan(rho) else 0.0


def compute_effective_rank(
    embeddings: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute effective rank (ERank) of embeddings.

    ERank = exp(entropy of normalized eigenvalues)

    Higher ERank indicates better dimensional utilization.

    Args:
        embeddings: Embedding matrix [N, D].
        eps: Numerical stability epsilon.

    Returns:
        Effective rank value.

    Notes:
        - Space: Spectral analysis of covariance
        - Status: Exact (via eigendecomposition)
        - ERank ∈ [1, D] where D is embedding dimension
    """
    try:
        # Center embeddings
        X = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = X.T @ X / (X.shape[0] - 1)

        # Add jitter for stability
        jitter = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype) * eps

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh((cov + jitter).float())
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)

        # Normalized eigenvalue distribution
        p = eigenvalues / eigenvalues.sum()

        # Shannon entropy
        entropy = -(p * torch.log(p)).sum()

        return torch.exp(entropy).item()
    except Exception:
        return 1.0


def compute_gromov_delta(
    dist_matrix: torch.Tensor,
    n_samples: int = 100,
) -> float:
    """
    Estimate Gromov δ-hyperbolicity.

    Lower δ indicates "more hyperbolic" structure.

    Args:
        dist_matrix: Pairwise distance matrix [N, N].
        n_samples: Number of 4-point samples.

    Returns:
        Estimated Gromov δ value.

    Notes:
        - Space: Four-point condition on metric space
        - Status: Monte Carlo estimate
        - δ = 0 for trees; δ > 0 for general spaces
    """
    n = dist_matrix.shape[0]
    if n < 4:
        return 0.0

    max_delta = 0.0
    d = dist_matrix

    for _ in range(n_samples):
        idx = torch.randperm(n, device=dist_matrix.device)[:4]
        x, y, z, w = idx[0], idx[1], idx[2], idx[3]

        # Gromov products
        gromov_xy_w = 0.5 * (d[x, w] + d[y, w] - d[x, y])
        gromov_xz_w = 0.5 * (d[x, w] + d[z, w] - d[x, z])
        gromov_yz_w = 0.5 * (d[y, w] + d[z, w] - d[y, z])

        # δ-hyperbolicity condition
        products = torch.stack([gromov_xy_w, gromov_xz_w, gromov_yz_w])
        sorted_prods, _ = products.sort()

        delta = (sorted_prods[1] - sorted_prods[0]).item()
        max_delta = max(max_delta, max(0, delta))

    return max_delta


def compute_distortion(
    D_original: torch.Tensor,
    D_compressed: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute distortion metrics between distance matrices.

    Args:
        D_original: Original distance matrix.
        D_compressed: Compressed distance matrix.

    Returns:
        Dictionary with distortion statistics.

    Notes:
        - Space: Distance ratio analysis
        - Status: Exact computation
        - Mean distortion ~1.0 indicates good preservation
    """
    # Normalize to [0, 1]
    D_orig_norm = D_original / (D_original.max() + 1e-8)
    D_comp_norm = D_compressed / (D_compressed.max() + 1e-8)

    # Exclude diagonal
    mask = ~torch.eye(D_original.shape[0], dtype=torch.bool, device=D_original.device)

    # Distortion ratio
    ratio = (D_comp_norm[mask] + 1e-8) / (D_orig_norm[mask] + 1e-8)

    # Correlation
    flat_orig = D_orig_norm[mask].flatten()
    flat_comp = D_comp_norm[mask].flatten()
    stacked = torch.stack([flat_orig, flat_comp])
    corr = torch.corrcoef(stacked)[0, 1].item()

    return {
        "distortion_mean": ratio.mean().item(),
        "distortion_std": ratio.std().item(),
        "distortion_max": ratio.max().item(),
        "correlation": corr if not np.isnan(corr) else 0.0,
    }
