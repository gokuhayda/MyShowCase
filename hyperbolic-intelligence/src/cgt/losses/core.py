# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Loss Functions
==================

Multi-objective loss functions for Contrastive Geometric Transfer training.

This module implements the complete loss landscape for CGT, including:
- Hyperbolic InfoNCE (contrastive alignment)
- Power-law distillation (distance compression)
- Spectral manifold alignment (eigenvalue preservation)
- Topological regularization (Betti-0 proxy)

Mathematical Status
-------------------
- Contrastive: Exact InfoNCE with geodesic distances
- Distillation: Empirical power-law scaling (NOT optimal transport)
- Spectral: First-order approximation via graph Laplacian
- Topological: Differentiable PROXY for connectivity (NOT global invariant)

Notes
-----
- All losses operate consistently on the Lorentz manifold
- Temperature annealing is heuristic, not theoretically optimal
- Loss weights are hyperparameters requiring validation

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicInfoNCE(nn.Module):
    """
    InfoNCE contrastive loss using hyperbolic geodesic distances.

    Unlike Euclidean contrastive losses that use cosine similarity,
    this loss operates directly on the manifold, ensuring metric
    consistency between training and evaluation.

    Similarity is computed as: sim(x,y) = -d_H(x,y) / τ

    Attributes:
        temperature: Scaling factor for logits (τ).

    Notes:
        - Space: Loss computed on manifold H^n
        - Status: Exact InfoNCE with geodesic distances
        - Ensures metric consistency: training uses same metric as evaluation
        - This CLOSES the tangent-space gap identified in prior hyperbolic work
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature scaling (τ). Lower = sharper distribution.

        Notes:
            - τ ∈ [0.05, 0.5] typical for sentence embeddings
            - Lower τ increases gradient magnitude
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,  # Unused but kept for interface compatibility
        substrate: nn.Module,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with geodesic distances.

        Args:
            student_emb: Hyperbolic embeddings [B, n+1].
            teacher_emb: Teacher embeddings (unused, for interface compatibility).
            substrate: Lorentz substrate for distance computation.

        Returns:
            InfoNCE loss (scalar).

        Notes:
            - Space: Computed on manifold H^n
            - Status: Exact contrastive loss
            - Diagonal elements are positive pairs (self-alignment)
        """
        B = student_emb.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=student_emb.device, requires_grad=True)

        # Geodesic distance matrix
        D = substrate.distance_matrix(student_emb)  # [B, B]

        # Similarity = negative distance (closer = higher similarity)
        logits = -D / self.temperature

        # Numerical stability: subtract max
        logits = logits - logits.max(dim=1, keepdim=True).values

        # Diagonal elements are positive pairs
        labels = torch.arange(B, device=student_emb.device)

        return F.cross_entropy(logits, labels)


class PowerLawDistillation(nn.Module):
    """
    Power-law distance distillation from Euclidean to hyperbolic space.

    Compresses teacher distances using: d_target = d_teacher^α

    This empirical scaling addresses the capacity mismatch between
    high-dimensional Euclidean space and compact hyperbolic space.

    Attributes:
        alpha: Power-law exponent (α ∈ (0, 1) for compression).

    Notes:
        - Space: Maps Euclidean distances to hyperbolic target distances
        - Status: EMPIRICAL heuristic (NOT optimal transport)
        - Does NOT claim OT-optimality or theoretical optimum
        - α = 0.5 chosen via grid search, not derived theoretically
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize distillation loss.

        Args:
            alpha: Power-law exponent. α < 1 compresses distances.

        Notes:
            - α = 0.5 is empirically effective for 24× compression
            - α = 1.0 would be identity (no compression)
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_normalized: torch.Tensor,
        substrate: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute power-law distillation loss.

        Args:
            student_emb: Hyperbolic embeddings [B, n+1].
            teacher_normalized: L2-normalized teacher embeddings [B, D].
            substrate: Lorentz substrate.

        Returns:
            Tuple of (loss, D_student, D_teacher_scaled, mean_teacher_dist).

        Notes:
            - Space: Distance comparison between spaces
            - Status: Empirical MSE loss (heuristic)
            - Both matrices normalized to [0,1] before comparison
        """
        # Teacher cosine distances (1 - cosine similarity)
        D_teacher = 1.0 - torch.mm(teacher_normalized, teacher_normalized.t())
        D_teacher = torch.clamp(D_teacher, min=1e-7)

        # Power-law scaling
        D_teacher_scaled = torch.pow(D_teacher, self.alpha)

        # Student geodesic distances
        D_student = substrate.distance_matrix(student_emb)

        # Normalize both to [0, 1] for comparison
        D_s_norm = D_student / (D_student.max() + 1e-8)
        D_t_norm = D_teacher_scaled / (D_teacher_scaled.max() + 1e-8)

        # MSE loss on normalized distance matrices
        loss = F.mse_loss(D_s_norm, D_t_norm.detach())

        return loss, D_student, D_teacher_scaled.detach(), D_teacher.mean().item()


class SpectralManifoldAlignment(nn.Module):
    """
    Spectral alignment loss via graph Laplacian eigenvalues.

    Encourages the student embedding to preserve the spectral
    properties of the teacher's similarity graph.

    Attributes:
        sigma: RBF kernel bandwidth.
        min_batch: Minimum batch size for computation.
        max_batch: Maximum batch size (memory constraint).

    Notes:
        - Space: Spectral comparison of similarity graphs
        - Status: First-order approximation via batch Laplacian
        - Uses top-k eigenvalues for efficiency
        - Does NOT guarantee global spectral preservation
    """

    def __init__(
        self,
        sigma: float = 1.0,
        min_batch: int = 16,
        max_batch: int = 128,
    ):
        """
        Initialize spectral alignment loss.

        Args:
            sigma: RBF kernel bandwidth for similarity graphs.
            min_batch: Skip computation below this batch size.
            max_batch: Subsample batches larger than this.
        """
        super().__init__()
        self.sigma = sigma
        self.min_batch = min_batch
        self.max_batch = max_batch

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        substrate: nn.Module,
    ) -> torch.Tensor:
        """
        Compute spectral alignment loss.

        Args:
            student_emb: Hyperbolic embeddings [B, n+1].
            teacher_emb: Teacher embeddings [B, D].
            substrate: Lorentz substrate.

        Returns:
            Spectral alignment loss (scalar).

        Notes:
            - Space: Eigenspace comparison
            - Status: First-order approximation (batch-local)
            - May subsample for large batches
        """
        B = student_emb.shape[0]
        device = student_emb.device
        dtype = student_emb.dtype

        if B < self.min_batch:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # Subsample if needed
        if B > self.max_batch:
            idx = torch.randperm(B, device=device)[: self.max_batch]
            student_emb = student_emb[idx]
            teacher_emb = teacher_emb[idx]
            B = self.max_batch

        # Teacher similarity via RBF kernel
        t_norm = F.normalize(teacher_emb, dim=-1)
        D_t = 1.0 - torch.mm(t_norm, t_norm.t())
        A_t = torch.exp(-D_t / (2 * self.sigma**2))

        # Student similarity via geodesic
        D_s = substrate.distance_matrix(student_emb)
        A_s = torch.exp(-D_s / (2 * self.sigma**2))

        # Graph Laplacians
        L_t = torch.diag(A_t.sum(dim=1)) - A_t
        L_s = torch.diag(A_s.sum(dim=1)) - A_s

        # Top-k eigenvalues (k = min(10, B))
        k = min(10, B)
        try:
            eig_t = torch.linalg.eigvalsh(L_t.float())[:k]
            eig_s = torch.linalg.eigvalsh(L_s.float())[:k]
            loss = F.mse_loss(eig_s.to(dtype), eig_t.to(dtype).detach())
        except RuntimeError:
            # Fallback for numerical issues
            loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return loss


class TopoLoss(nn.Module):
    """
    Topological regularization via Betti-0 proxy.

    Uses spectral properties of the distance-based graph Laplacian
    as a DIFFERENTIABLE PROXY for connectivity (Betti-0).

    Attributes:
        target_beta_0: Target connectivity value.
        temp_init: Initial temperature for soft thresholding.
        temp_min: Minimum temperature (annealing floor).

    Notes:
        - Space: Spectral proxy for topological connectivity
        - Status: Differentiable PROXY (NOT exact Betti number)
        - Does NOT guarantee global homology preservation
        - Temperature annealing improves convergence but is HEURISTIC
        - Computational complexity: O(B³) due to eigendecomposition

    Limitations:
        - Only captures local connectivity (batch-level)
        - Not a topological invariant in the mathematical sense
        - May conflict with other objectives during training
    """

    def __init__(
        self,
        target_beta_0: float = 1.0,
        temp_init: float = 0.2,
        temp_min: float = 0.03,
    ):
        """
        Initialize topological loss.

        Args:
            target_beta_0: Target number of connected components.
            temp_init: Starting temperature for annealing.
            temp_min: Floor temperature.

        Notes:
            - target_beta_0 = 1.0 encourages single connected component
            - Temperature controls soft threshold sharpness
        """
        super().__init__()
        self.target = target_beta_0
        self.temp_init = temp_init
        self.temp_min = temp_min

    def temperature(self, epoch: int, max_epoch: int) -> float:
        """
        Compute annealed temperature.

        Args:
            epoch: Current epoch.
            max_epoch: Total epochs.

        Returns:
            Current temperature value.

        Notes:
            - Linear annealing from temp_init to temp_min
            - Status: Heuristic schedule (not theoretically optimal)
        """
        return max(
            self.temp_min, self.temp_init * (1 - epoch / max_epoch)
        )

    def forward(
        self,
        D: torch.Tensor,
        epoch: int = 0,
        max_epoch: int = 100,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute topological regularization loss.

        Args:
            D: Distance matrix [B, B].
            epoch: Current training epoch.
            max_epoch: Maximum epochs for annealing.

        Returns:
            Tuple of (loss, estimated_beta_0).

        Notes:
            - Space: Spectral analysis of distance graph
            - Status: Differentiable proxy (O(B³) complexity)
            - beta_0 estimate via exponential eigenvalue sum
        """
        tau = self.temperature(epoch, max_epoch)

        # Soft adjacency via sigmoid threshold
        adj = torch.sigmoid((2.0 - D) / tau)

        # Graph Laplacian
        laplacian = torch.diag(adj.sum(dim=1)) - adj

        # Eigenvalues
        try:
            eigs = torch.linalg.eigvalsh(laplacian.float())
        except RuntimeError:
            return torch.tensor(0.0, device=D.device, requires_grad=True), 0.0

        # Soft Betti-0 estimate
        beta_0 = torch.sum(torch.exp(-eigs / tau))

        loss = (beta_0 - self.target) ** 2

        return loss, beta_0.item()


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss aggregator for CGT training.

    Combines all loss components with configurable weights:
    - Contrastive (InfoNCE with geodesic distances)
    - Distillation (power-law distance compression)
    - Spectral (Laplacian eigenvalue alignment)
    - Topological (Betti-0 proxy regularization)
    - Lipschitz (metric continuity regularization)

    Attributes:
        lambda_*: Weight coefficients for each loss component.
        temperature: InfoNCE temperature.
        target_beta_0: Target connectivity.
        power_law_alpha: Distillation compression exponent.

    Notes:
        - Space: Aggregated loss on manifold H^n
        - Status: Weighted sum (linear scalarization)
        - Weight tuning is empirical; no Pareto optimality guarantees
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.7,
        lambda_spectral: float = 0.2,
        lambda_topo: float = 0.02,
        lambda_lipschitz: float = 0.005,
        radius_weight: float = 0.05,
        temperature: float = 0.07,
        target_beta_0: float = 1.0,
        power_law_alpha: float = 0.5,
    ):
        """
        Initialize multi-objective loss.

        Args:
            lambda_contrastive: Weight for contrastive loss.
            lambda_distill: Weight for distillation loss.
            lambda_spectral: Weight for spectral alignment.
            lambda_topo: Weight for topological regularization.
            lambda_lipschitz: Weight for Lipschitz regularization.
            radius_weight: Weight for radius regularization.
            temperature: InfoNCE temperature.
            target_beta_0: Target Betti-0 value.
            power_law_alpha: Power-law exponent for distillation.
        """
        super().__init__()

        # Store weights
        self.lambda_contrastive = lambda_contrastive
        self.lambda_distill = lambda_distill
        self.lambda_spectral = lambda_spectral
        self.lambda_topo = lambda_topo
        self.lambda_lipschitz = lambda_lipschitz
        self.radius_weight = radius_weight

        # Initialize component losses
        self.contrastive = HyperbolicInfoNCE(temperature)
        self.distillation = PowerLawDistillation(power_law_alpha)
        self.spectral = SpectralManifoldAlignment()
        self.topo = TopoLoss(target_beta_0)

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        substrate: nn.Module,
        model: nn.Module,
        epoch: int = 0,
        max_epoch: int = 100,
        lipschitz_reg: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and component breakdown.

        Args:
            student_emb: Hyperbolic embeddings [B, n+1].
            teacher_emb: Teacher embeddings [B, D].
            substrate: Lorentz substrate.
            model: Student model (for Lipschitz computation).
            epoch: Current epoch.
            max_epoch: Total epochs.
            lipschitz_reg: Optional Lipschitz regularizer.

        Returns:
            Tuple of (total_loss, loss_dict).

        Notes:
            - Space: Combined loss on manifold
            - Status: Linear scalarization of objectives
        """
        device = student_emb.device
        dtype = student_emb.dtype
        t_norm = F.normalize(teacher_emb, dim=-1)

        # Component losses
        L_contrastive = self.contrastive(student_emb, teacher_emb, substrate)
        L_distill, D_s, D_t, _ = self.distillation(student_emb, t_norm, substrate)
        L_spectral = self.spectral(student_emb, teacher_emb, substrate)
        L_topo, beta_0 = self.topo(D_s, epoch, max_epoch)

        # Lipschitz regularization (optional)
        if lipschitz_reg is not None:
            L_lipschitz = lipschitz_reg(model, teacher_emb)
        else:
            L_lipschitz = torch.tensor(0.0, device=device, dtype=dtype)

        # Radius regularization
        radii = substrate.lorentz_radius(student_emb)
        L_radius = F.relu(radii - 10.0).mean()  # Penalize radius > 10

        # Total weighted loss
        total = (
            self.lambda_contrastive * L_contrastive
            + self.lambda_distill * L_distill
            + self.lambda_spectral * L_spectral
            + self.lambda_topo * L_topo
            + self.lambda_lipschitz * L_lipschitz
            + self.radius_weight * L_radius
        )

        # Return dict for logging
        loss_dict = {
            "total": total.item(),
            "contrastive": L_contrastive.item(),
            "distill": L_distill.item(),
            "spectral": L_spectral.item(),
            "topo": L_topo.item(),
            "lipschitz": L_lipschitz.item(),
            "radius": L_radius.item(),
            "beta_0": beta_0,
        }

        return total, loss_dict
