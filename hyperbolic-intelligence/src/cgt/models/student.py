# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Model Architecture
======================

Neural network components for Contrastive Geometric Transfer.

This module implements the CGT student encoder that projects
high-dimensional Euclidean embeddings into compact hyperbolic space.

Architecture
------------
1. Euclidean Projector: MLP with spectral normalization
2. Tangent Vector Construction: Lift to tangent space at origin
3. Exponential Map: Project onto Lorentz hyperboloid
4. Homeostatic Field (optional): Anchor-based density preservation

Mathematical Status
-------------------
- Projector: Standard MLP (no manifold structure)
- Exponential map: Exact closed-form
- Homeostatic field: Heuristic density regularization

Notes
-----
- Spectral normalization ensures bounded Lipschitz constant per layer
- Homeostatic anchors are learnable parameters on the manifold
- Forward pass maintains manifold membership via projection

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from cgt.geometry import LorentzConfig, LorentzSubstrate


def create_projector(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_spectral: bool = True,
) -> nn.Sequential:
    """
    Create Euclidean projector backbone with optional spectral normalization.

    Architecture: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear

    Args:
        input_dim: Input dimension (teacher embedding size).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (hyperbolic intrinsic dimension).
        use_spectral: Apply spectral normalization to linear layers.

    Returns:
        Sequential module implementing the projector.

    Notes:
        - Space: Euclidean MLP (pre-manifold)
        - Status: Standard neural network with optional spectral constraint
        - Spectral norm bounds operator norm ≤ 1 per layer
        - Orthogonal initialization promotes dimensional spread
    """

    def maybe_sn(layer: nn.Linear) -> nn.Module:
        return spectral_norm(layer) if use_spectral else layer

    l1 = nn.Linear(input_dim, hidden_dim)
    l2 = nn.Linear(hidden_dim, hidden_dim)
    l3 = nn.Linear(hidden_dim, output_dim)

    # Orthogonal initialization for dimensional diversity
    nn.init.orthogonal_(l1.weight, gain=1.0)
    nn.init.orthogonal_(l2.weight, gain=1.0)

    # Small initialization for output to prevent exp_map explosion
    nn.init.normal_(l3.weight, std=1e-4)
    if l3.bias is not None:
        nn.init.zeros_(l3.bias)

    return nn.Sequential(
        maybe_sn(l1),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        maybe_sn(l2),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        maybe_sn(l3),
    )


class HomeostaticField(nn.Module):
    """
    Homeostatic field for density and connectivity preservation.

    Uses learnable anchor points on the manifold to prevent
    embedding collapse and maintain spatial diversity.

    During training, points are gently attracted toward nearby anchors,
    encouraging uniform coverage of the embedding space.

    Attributes:
        substrate: Lorentz substrate for geometric operations.
        n_anchors: Number of anchor points.
        alpha: Attraction strength (0 = disabled, 1 = full attraction).
        anchors: Learnable anchor parameters on manifold.

    Notes:
        - Space: Operates on manifold H^n
        - Status: Heuristic regularization (not theoretically grounded)
        - Anchors have on_manifold=True flag for Riemannian optimization
        - Only active during training (identity at eval time)
    """

    def __init__(
        self,
        substrate: LorentzSubstrate,
        n_anchors: int = 16,
        alpha: float = 0.1,
    ):
        """
        Initialize homeostatic field.

        Args:
            substrate: Lorentz substrate.
            n_anchors: Number of learnable anchor points.
            alpha: Attraction coefficient (lower = weaker effect).

        Notes:
            - Anchors initialized near origin for stability
            - on_manifold flag enables Riemannian gradient descent
        """
        super().__init__()
        self.substrate = substrate
        self.n_anchors = n_anchors
        self.alpha = alpha

        # Initialize anchors near origin
        s_dim = substrate.n
        init_spatial = torch.randn(n_anchors, s_dim) * 0.01

        # Compute time coordinate for manifold membership
        init_time = torch.sqrt(
            1.0 + (init_spatial**2).sum(dim=1, keepdim=True)
        )
        init_points = torch.cat([init_time, init_spatial], dim=-1)

        self.anchors = nn.Parameter(init_points)
        self.anchors.on_manifold = True  # Flag for Riemannian optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply homeostatic field during training.

        Args:
            x: Points on manifold [B, n+1].

        Returns:
            Adjusted points [B, n+1].

        Notes:
            - Space: Adjustment on manifold via tangent vector
            - Status: Heuristic drift toward anchors
            - Identity at evaluation time
        """
        if not self.training:
            return x

        device = x.device
        dtype = x.dtype

        # Ensure anchors are on manifold
        anchors_fixed = self.substrate.proj(self.anchors.to(device, dtype))

        # Find nearest anchor for each point
        dists = self.substrate.distance_matrix_points(x, anchors_fixed)
        nearest_idx = dists.argmin(dim=1)
        nearest_anchors = anchors_fixed[nearest_idx]

        # Compute drift direction in tangent space
        direction = self.substrate.log_map(x, nearest_anchors)

        # Apply scaled drift via exponential map
        return self.substrate.exp_map(x, self.alpha * direction)


class CGTStudent(nn.Module):
    """
    Contrastive Geometric Transfer Student Encoder.

    Projects high-dimensional Euclidean teacher embeddings into
    compact hyperbolic (Lorentz) space for efficient retrieval.

    Pipeline:
    1. teacher_emb ∈ R^D (e.g., 768-dim SBERT)
    2. projected = Projector(teacher_emb) ∈ R^n (e.g., 32-dim)
    3. tangent = [0, projected] ∈ T_o H^n
    4. hyp_emb = exp_o(tangent) ∈ H^n (33-dim Lorentz coords)
    5. (optional) hyp_emb = HomestaticField(hyp_emb)

    Attributes:
        student_dim: Intrinsic hyperbolic dimension.
        substrate: Lorentz manifold substrate.
        projector: Euclidean MLP backbone.
        scale: Tangent vector scaling factor.
        homeostatic_layer: Optional homeostatic field.
        anchors: Reference to homeostatic anchors (for optimizer).

    Notes:
        - Space: Maps R^D → H^n
        - Status: Exact exponential map; MLP is standard
        - Achieves 24× compression (768 → 32)
        - Output dimension is n+1 due to Lorentz time coordinate
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        hidden_dim: int = 256,
        learnable_curvature: bool = True,
        initial_curvature: float = 1.0,
        curvature_min: float = 0.1,
        curvature_max: float = 5.0,
    ):
        """
        Initialize CGT student encoder.

        Args:
            teacher_dim: Teacher embedding dimension (e.g., 768).
            student_dim: Hyperbolic intrinsic dimension (e.g., 32).
            hidden_dim: Projector hidden dimension.
            learnable_curvature: Enable curvature learning.
            initial_curvature: Starting curvature value.
            curvature_min: Minimum curvature bound.
            curvature_max: Maximum curvature bound.

        Notes:
            - Output has dimension student_dim + 1 (Lorentz coords)
            - Compression ratio = teacher_dim / student_dim
        """
        super().__init__()

        config = LorentzConfig(
            intrinsic_dim=student_dim,
            initial_curvature=initial_curvature,
            learnable_curvature=learnable_curvature,
            curvature_min=curvature_min,
            curvature_max=curvature_max,
        )

        self.student_dim = student_dim
        self.substrate = LorentzSubstrate(config)
        self.projector = create_projector(
            teacher_dim, hidden_dim, student_dim, use_spectral=True
        )

        # Scale factor for tangent vector magnitude
        self.register_buffer("scale", torch.tensor(0.7))

        # Placeholders for homeostatic components
        self.homeostatic_layer: Optional[HomeostaticField] = None
        self.anchors: Optional[nn.Parameter] = None

    def init_homeostatic(
        self,
        n_anchors: int = 16,
        alpha: float = 0.1,
    ) -> HomeostaticField:
        """
        Initialize homeostatic field.

        Args:
            n_anchors: Number of anchor points.
            alpha: Attraction strength.

        Returns:
            Initialized HomeostaticField module.

        Notes:
            - Anchors exposed at model level for Riemannian optimizer
            - on_manifold flag enables specialized gradient handling
        """
        self.homeostatic_layer = HomeostaticField(
            self.substrate, n_anchors, alpha
        )

        # Expose anchors at model level
        self.anchors = self.homeostatic_layer.anchors
        self.anchors.on_manifold = True

        return self.homeostatic_layer

    def forward(
        self,
        teacher_emb: torch.Tensor,
        use_homeostatic: bool = True,
    ) -> torch.Tensor:
        """
        Encode teacher embeddings into hyperbolic space.

        Args:
            teacher_emb: Teacher embeddings [B, D].
            use_homeostatic: Apply homeostatic field if available.

        Returns:
            Hyperbolic embeddings [B, n+1] on Lorentz manifold.

        Notes:
            - Space: Output lives on H^n ⊂ R^{n+1}
            - Status: Exact exponential map application
            - Manifold membership guaranteed via projection
        """
        device = teacher_emb.device
        dtype = teacher_emb.dtype
        batch_size = teacher_emb.shape[0]

        # Ensure model on correct device
        if next(self.projector.parameters()).device != device:
            self.to(device)

        # 1. Euclidean projection
        projected = self.projector(teacher_emb)

        # 2. Normalize and scale
        projected = F.normalize(projected, dim=-1) * self.scale

        # 3. Construct tangent vector at origin: [0, v_1, ..., v_n]
        tangent = torch.zeros(
            batch_size, self.student_dim + 1, device=device, dtype=dtype
        )
        tangent[:, 1:] = projected

        # 4. Map to manifold via exponential map
        origin = self.substrate.origin(batch_size).to(device, dtype)
        hyp_emb = self.substrate.exp_map(origin, tangent)

        # 5. Optional homeostatic refinement
        if use_homeostatic and self.homeostatic_layer is not None:
            hyp_emb = self.homeostatic_layer(hyp_emb)

        return hyp_emb

    def get_curvature(self) -> float:
        """
        Get current curvature value.

        Returns:
            Curvature parameter K.
        """
        return self.substrate.K.item()

    def get_radius_stats(self, embeddings: torch.Tensor) -> dict:
        """
        Compute radius statistics for embeddings.

        Args:
            embeddings: Hyperbolic embeddings [B, n+1].

        Returns:
            Dictionary with mean, std, min, max radius.

        Notes:
            - Radius = geodesic distance from origin
            - Useful for monitoring embedding spread
        """
        radii = self.substrate.lorentz_radius(embeddings)
        return {
            "mean": radii.mean().item(),
            "std": radii.std().item(),
            "min": radii.min().item(),
            "max": radii.max().item(),
        }


class RiemannianOptimizerWrapper:
    """
    Wrapper for PyTorch optimizers with Riemannian support.

    Performs standard updates for Euclidean parameters and
    exponential map retraction for parameters marked with
    on_manifold=True.

    Attributes:
        optimizer: Base PyTorch optimizer.
        substrate: Lorentz substrate for manifold operations.

    Notes:
        - Space: Hybrid Euclidean/Riemannian optimization
        - Status: Exact exponential map retraction
        - Parameters with on_manifold flag get Riemannian treatment
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        substrate: LorentzSubstrate,
    ):
        """
        Initialize Riemannian optimizer wrapper.

        Args:
            optimizer: Base optimizer (e.g., AdamW).
            substrate: Lorentz substrate for geometry.
        """
        self.optimizer = optimizer
        self.substrate = substrate

    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform hybrid optimization step.

        1. Standard update for Euclidean parameters
        2. Riemannian retraction for manifold parameters

        Notes:
            - Euclidean params: standard gradient descent
            - Manifold params: exp_x(-lr * riemannian_grad)
        """
        # Standard step for all parameters
        self.optimizer.step()

        # Riemannian retraction for manifold parameters
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if getattr(p, "on_manifold", False) and p.grad is not None:
                        # Convert to Riemannian gradient
                        r_grad = self.substrate.riemannian_grad(p.data, p.grad.data)

                        # Retraction via exponential map
                        p.data = self.substrate.exp_map(p.data, -lr * r_grad)

                        # Paranoid projection
                        p.data = self.substrate.proj(p.data)

    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
