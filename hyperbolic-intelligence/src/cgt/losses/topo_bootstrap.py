# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Bootstrap-Stabilized Topological Loss
=====================================

Addresses the critical gap identified in CGT/HGST audit regarding
instability of Betti numbers computed on mini-batches.

The Problem:
    Mini-batch sampling creates artificial fragmentation of the data manifold,
    leading to inflated Betti-0 (connected components) that doesn't reflect
    the true topology of the full dataset.

The Solution:
    Use Smoothed Bootstrap to estimate confidence intervals for Betti numbers,
    penalizing only topology changes that are statistically significant
    outside the sampling noise.

Mathematical Status
-------------------
- Betti Computation: PROXY (spectral Laplacian, not exact persistent homology)
- Bootstrap: STATISTICAL (empirically validated)
- Confidence Intervals: APPROXIMATION (assumes asymptotic normality)

For exact topological computation, integrate with PETLS or GUDHI.

References:
    - Fasy et al. "Confidence sets for persistence diagrams"
    - PETLS: PErsistent Topological Laplacian Software

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def safe_sqrt(x: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """Numerically stable square root."""
    return torch.sqrt(torch.clamp(x, min=eps))


@dataclass
class BettiEstimate:
    """
    Bootstrap estimate of Betti number with confidence interval.
    
    Attributes:
        mean: Point estimate of Betti number
        std: Standard deviation from bootstrap
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        n_bootstrap: Number of bootstrap samples used
    """
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    
    def is_significant_change(self, other: 'BettiEstimate', alpha: float = 0.05) -> bool:
        """
        Test if two Betti estimates are significantly different.
        
        Uses overlap of confidence intervals as a conservative test.
        """
        # No overlap = significant difference
        return self.ci_upper < other.ci_lower or other.ci_upper < self.ci_lower


class SpectralBettiEstimator(nn.Module):
    """
    Spectral proxy for Betti-0 (connected components).
    
    Uses the nullity (dimension of kernel) of the graph Laplacian
    as a proxy for Betti-0. The number of zero eigenvalues equals
    the number of connected components.
    
    IMPORTANT: This is a PROXY, not exact persistent homology.
    For publication-grade results, integrate with PETLS or GUDHI.
    
    Args:
        threshold: Eigenvalue threshold for "zero" (numerical tolerance)
        sigma_scale: Scale factor for adjacency kernel bandwidth
        
    Notes:
        - Mathematical Identity: Spectral Graph Theory
        - Classification: PROXY (not exact Betti computation)
        - Limitation: Sensitive to kernel bandwidth and threshold
    """
    
    def __init__(
        self,
        threshold: float = 0.01,
        sigma_scale: float = 1.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.sigma_scale = sigma_scale
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Estimate Betti-0 from embeddings via spectral proxy.
        
        Args:
            embeddings: Point cloud [N, dim]
            
        Returns:
            Estimated Betti-0 (soft, differentiable)
        """
        n = embeddings.shape[0]
        device = embeddings.device
        dtype = embeddings.dtype
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)
        
        # Adaptive bandwidth
        sigma = self.sigma_scale * dists.mean()
        
        # Gaussian kernel adjacency
        adj = torch.exp(-dists ** 2 / (2 * sigma ** 2))
        
        # Remove self-loops for Laplacian
        adj = adj - torch.eye(n, device=device, dtype=dtype)
        adj = torch.clamp(adj, min=0.0)
        
        # Degree matrix
        degree = adj.sum(dim=-1)
        
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / safe_sqrt(degree + 1e-8))
        L = torch.eye(n, device=device, dtype=dtype) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Eigenvalues
        try:
            eigvals = torch.linalg.eigvalsh(L)
        except:
            # Fallback for numerical issues
            return torch.tensor(1.0, device=device, dtype=dtype)
        
        # Count "zero" eigenvalues (soft, differentiable)
        # Using sigmoid for smooth approximation
        soft_zeros = torch.sigmoid((self.threshold - eigvals) / 0.001)
        betti_0 = soft_zeros.sum()
        
        return betti_0
    
    def hard_betti_0(self, embeddings: torch.Tensor) -> int:
        """
        Compute hard (non-differentiable) Betti-0 estimate.
        
        More accurate but not suitable for training.
        """
        with torch.no_grad():
            n = embeddings.shape[0]
            device = embeddings.device
            dtype = embeddings.dtype
            
            dists = torch.cdist(embeddings, embeddings)
            sigma = self.sigma_scale * dists.mean()
            adj = torch.exp(-dists ** 2 / (2 * sigma ** 2))
            adj = adj - torch.eye(n, device=device, dtype=dtype)
            adj = torch.clamp(adj, min=0.0)
            
            degree = adj.sum(dim=-1)
            D_inv_sqrt = torch.diag(1.0 / safe_sqrt(degree + 1e-8))
            L = torch.eye(n, device=device, dtype=dtype) - D_inv_sqrt @ adj @ D_inv_sqrt
            
            eigvals = torch.linalg.eigvalsh(L)
            
            # Hard count
            return (eigvals < self.threshold).sum().item()


class BootstrapBettiEstimator(nn.Module):
    """
    Bootstrap-stabilized Betti number estimation.
    
    Addresses the mini-batch instability problem by:
    1. Sampling multiple bootstrap replicates from the batch
    2. Computing Betti-0 for each replicate
    3. Estimating confidence intervals
    4. Penalizing only significant topology changes
    
    Args:
        n_bootstrap: Number of bootstrap samples
        subsample_ratio: Fraction of points per bootstrap sample
        confidence: Confidence level for intervals
        betti_estimator: Underlying Betti estimator (default: spectral)
        
    Notes:
        - Mathematical Identity: Bootstrap Confidence Intervals
        - Classification: STATISTICAL (asymptotically valid)
        - Proof Status: Efron & Tibshirani (1993)
    """
    
    def __init__(
        self,
        n_bootstrap: int = 50,
        subsample_ratio: float = 0.8,
        confidence: float = 0.95,
        betti_estimator: Optional[SpectralBettiEstimator] = None,
    ):
        super().__init__()
        self.n_bootstrap = n_bootstrap
        self.subsample_ratio = subsample_ratio
        self.confidence = confidence
        self.betti_estimator = betti_estimator or SpectralBettiEstimator()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        return_ci: bool = False,
    ) -> torch.Tensor:
        """
        Compute bootstrap-stabilized Betti-0 estimate.
        
        Args:
            embeddings: Point cloud [N, dim]
            return_ci: Whether to return confidence interval info
            
        Returns:
            Betti-0 estimate (mean of bootstrap samples)
        """
        n = embeddings.shape[0]
        subsample_size = max(int(n * self.subsample_ratio), 10)
        
        betti_samples = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = torch.randint(0, n, (subsample_size,), device=embeddings.device)
            subsample = embeddings[idx]
            
            # Estimate Betti-0
            betti = self.betti_estimator(subsample)
            betti_samples.append(betti)
        
        betti_stack = torch.stack(betti_samples)
        mean_betti = betti_stack.mean()
        
        if return_ci:
            std_betti = betti_stack.std()
            # Percentile confidence interval
            alpha = 1 - self.confidence
            ci_lower = torch.quantile(betti_stack, alpha / 2)
            ci_upper = torch.quantile(betti_stack, 1 - alpha / 2)
            
            return mean_betti, std_betti, ci_lower, ci_upper
        
        return mean_betti
    
    def estimate_with_ci(self, embeddings: torch.Tensor) -> BettiEstimate:
        """
        Get full BettiEstimate with confidence interval.
        
        Non-differentiable, for analysis only.
        """
        with torch.no_grad():
            mean, std, ci_lower, ci_upper = self.forward(embeddings, return_ci=True)
            
            return BettiEstimate(
                mean=mean.item(),
                std=std.item(),
                ci_lower=ci_lower.item(),
                ci_upper=ci_upper.item(),
                n_bootstrap=self.n_bootstrap,
            )


class BootstrapTopoLoss(nn.Module):
    """
    Bootstrap-stabilized topological loss.
    
    Only penalizes topology changes that are statistically significant,
    avoiding the false positives caused by mini-batch sampling noise.
    
    Loss = λ * max(0, |Betti_observed - Betti_target| - margin)²
    
    where margin is estimated from bootstrap confidence intervals.
    
    Args:
        target_betti_0: Target number of connected components
        n_bootstrap: Number of bootstrap samples
        subsample_ratio: Fraction of points per bootstrap
        confidence: Confidence level for significance testing
        penalty_weight: Weight of the loss term
        
    Notes:
        - Mathematical Identity: Bootstrapped Topology Preservation
        - Classification: STATISTICAL + PROXY
        - Improvement over naive TopoLoss: Accounts for sampling variance
    """
    
    def __init__(
        self,
        target_betti_0: float = 1.0,
        n_bootstrap: int = 30,
        subsample_ratio: float = 0.8,
        confidence: float = 0.90,
        penalty_weight: float = 0.05,
    ):
        super().__init__()
        self.target_betti_0 = target_betti_0
        self.penalty_weight = penalty_weight
        self.confidence = confidence
        
        self.bootstrap_estimator = BootstrapBettiEstimator(
            n_bootstrap=n_bootstrap,
            subsample_ratio=subsample_ratio,
            confidence=confidence,
        )
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute bootstrap-stabilized topology loss.
        
        Args:
            embeddings: Point cloud [N, dim]
            
        Returns:
            Tuple of (loss, info_dict)
        """
        # Get bootstrap estimate with confidence interval
        mean_betti, std_betti, ci_lower, ci_upper = self.bootstrap_estimator(
            embeddings, return_ci=True
        )
        
        # Margin based on confidence interval width
        margin = (ci_upper - ci_lower) / 2
        
        # Deviation from target
        deviation = torch.abs(mean_betti - self.target_betti_0)
        
        # Only penalize if deviation exceeds margin (statistically significant)
        significant_deviation = torch.clamp(deviation - margin, min=0.0)
        
        loss = self.penalty_weight * (significant_deviation ** 2)
        
        info = {
            'betti_0_mean': mean_betti.item(),
            'betti_0_std': std_betti.item(),
            'betti_0_ci_lower': ci_lower.item(),
            'betti_0_ci_upper': ci_upper.item(),
            'margin': margin.item(),
            'significant': (deviation > margin).item(),
        }
        
        return loss, info


class AdaptiveTopoLoss(nn.Module):
    """
    Adaptive topological loss with automatic target estimation.
    
    Instead of requiring a fixed target Betti-0, this loss:
    1. Estimates the "natural" Betti-0 from teacher embeddings
    2. Penalizes significant deviations from teacher topology
    
    This is more robust for distillation scenarios where the
    target topology is not known a priori.
    
    Args:
        n_bootstrap: Number of bootstrap samples
        penalty_weight: Weight of the loss term
        warmup_batches: Number of batches to estimate teacher topology
        
    Notes:
        - Classification: ADAPTIVE + STATISTICAL
        - Use Case: Knowledge distillation with topology preservation
    """
    
    def __init__(
        self,
        n_bootstrap: int = 30,
        penalty_weight: float = 0.05,
        warmup_batches: int = 10,
    ):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.warmup_batches = warmup_batches
        
        self.bootstrap_estimator = BootstrapBettiEstimator(n_bootstrap=n_bootstrap)
        
        # Running estimate of teacher topology
        self.register_buffer('teacher_betti_mean', torch.tensor(1.0))
        self.register_buffer('teacher_betti_std', torch.tensor(0.5))
        self.register_buffer('n_updates', torch.tensor(0))
    
    def update_teacher_estimate(self, teacher_embeddings: torch.Tensor):
        """
        Update running estimate of teacher topology.
        
        Call this during warmup phase with teacher embeddings.
        """
        with torch.no_grad():
            estimate = self.bootstrap_estimator.estimate_with_ci(teacher_embeddings)
            
            # Exponential moving average
            alpha = 0.1
            self.teacher_betti_mean = (
                (1 - alpha) * self.teacher_betti_mean + 
                alpha * estimate.mean
            )
            self.teacher_betti_std = (
                (1 - alpha) * self.teacher_betti_std +
                alpha * estimate.std
            )
            self.n_updates += 1
    
    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive topology loss.
        
        Args:
            student_embeddings: Student model embeddings
            teacher_embeddings: Optional teacher embeddings for warmup
            
        Returns:
            Tuple of (loss, info_dict)
        """
        # Update teacher estimate during warmup
        if teacher_embeddings is not None and self.n_updates < self.warmup_batches:
            self.update_teacher_estimate(teacher_embeddings)
        
        # Estimate student topology
        mean_betti, std_betti, ci_lower, ci_upper = self.bootstrap_estimator(
            student_embeddings, return_ci=True
        )
        
        # Margin based on combined uncertainty
        combined_std = safe_sqrt(std_betti ** 2 + self.teacher_betti_std ** 2)
        margin = 2.0 * combined_std  # ~95% confidence
        
        # Deviation from teacher
        deviation = torch.abs(mean_betti - self.teacher_betti_mean)
        
        # Only penalize significant deviations
        significant_deviation = torch.clamp(deviation - margin, min=0.0)
        
        loss = self.penalty_weight * (significant_deviation ** 2)
        
        info = {
            'student_betti_0': mean_betti.item(),
            'teacher_betti_0': self.teacher_betti_mean.item(),
            'margin': margin.item(),
            'significant': (deviation > margin).item(),
            'warmup_complete': (self.n_updates >= self.warmup_batches).item(),
        }
        
        return loss, info
