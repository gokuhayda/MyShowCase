# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Lorentz Substrate for Hyperbolic Geometry
=========================================

This module implements the Lorentz (hyperboloid) model of hyperbolic space,
providing the geometric foundation for Contrastive Geometric Transfer (CGT).

The Lorentz model embeds H^n as the upper sheet of a two-sheeted hyperboloid
in (n+1)-dimensional Minkowski space with signature (-,+,+,...,+).

Mathematical Status
-------------------
- All operations are EXACT closed-form expressions for constant negative curvature.
- Numerical corrections (paranoid projections) are explicitly labeled as such.

Notes
-----
- Space: Lorentz hyperboloid H^n embedded in Minkowski space R^{n,1}
- Curvature: K > 0 corresponds to sectional curvature -1/K in hyperbolic space
- Status: Core geometric substrate with exact closed-form operations

References
----------
- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations"
- Ganea et al. (2018). "Hyperbolic Neural Networks"
- Law et al. (2019). "Lorentzian Distance Learning for Hyperbolic Representations"

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LorentzConfig:
    """
    Configuration for the Lorentz (hyperbolic) manifold substrate.

    The Lorentz model embeds points on the upper sheet of a two-sheeted
    hyperboloid in Minkowski space with signature (-,+,+,...,+).

    Attributes:
        intrinsic_dim: Dimension of the hyperbolic space (n in H^n).
        eps: Numerical stability epsilon for clamp operations.
        learnable_curvature: Whether K is a learnable parameter.
        initial_curvature: Initial value of sectional curvature parameter K.
        curvature_min: Minimum allowed curvature (prevents manifold collapse).
        curvature_max: Maximum allowed curvature (prevents gradient explosion).

    Notes:
        - Space: Configuration for ambient Minkowski space R^{n,1}
        - Status: Exact parameter specification
        - The actual sectional curvature is -1/K
    """

    intrinsic_dim: int = 32
    eps: float = 1e-6
    learnable_curvature: bool = True
    initial_curvature: float = 1.0
    curvature_min: float = 0.1
    curvature_max: float = 10.0


def safe_acosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Numerically stable inverse hyperbolic cosine.

    Uses Taylor expansion near x=1 to avoid gradient explosion.

    Args:
        x: Input tensor (values should be >= 1).
        eps: Numerical stability epsilon.

    Returns:
        acosh(x) with stable gradients near x=1.

    Notes:
        - Space: Operates on scalar/tensor values in R
        - Status: Exact implementation with numerical stabilization
        - Uses identity: acosh(x) = log(x + sqrt(x^2 - 1)) for |x| > 1 + eps
        - Uses Taylor: acosh(1+h) ≈ sqrt(2h) for h small
    """
    x_safe = torch.clamp(x, min=1.0 + eps)
    near_one_mask = x_safe < 1.0 + 0.001

    # Standard formula for values away from 1
    result = torch.log(x_safe + torch.sqrt(x_safe**2 - 1.0 + eps))

    # Taylor approximation near 1: acosh(1+h) ≈ sqrt(2h)
    h = x_safe - 1.0
    taylor_approx = torch.sqrt(2.0 * h + eps)

    return torch.where(near_one_mask, taylor_approx, result)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """
    Protected square root to avoid infinite gradients at zero.

    Args:
        x: Input tensor.
        eps: Minimum clamp value.

    Returns:
        sqrt(max(x, eps))

    Notes:
        - Space: Operates on scalar/tensor values in R
        - Status: Exact with numerical floor
    """
    return torch.sqrt(torch.clamp(x, min=eps))


class LorentzSubstrate(nn.Module):
    """
    Lorentz (hyperboloid) model of hyperbolic space.

    Implements the upper sheet of the hyperboloid {x ∈ R^{n+1} : <x,x>_L = -1/K}
    embedded in Minkowski space with inner product <x,y>_L = -x₀y₀ + Σᵢxᵢyᵢ.

    This substrate provides all geometric operations needed for hyperbolic
    neural networks: inner products, distances, exponential/logarithmic maps,
    parallel transport, and manifold projections.

    Attributes:
        config: LorentzConfig specifying manifold parameters.
        n: Intrinsic dimension of hyperbolic space.
        eps: Numerical stability epsilon.

    Notes:
        - Space: Lorentz hyperboloid H^n ⊂ R^{n,1}
        - Status: Exact closed-form operations throughout
        - The time coordinate x₀ is always positive (upper sheet)
        - Sectional curvature is -1/K where K is the curvature parameter
    """

    def __init__(self, config: Optional[LorentzConfig] = None):
        """
        Initialize the Lorentz substrate.

        Args:
            config: Manifold configuration. Uses defaults if None.

        Notes:
            - Curvature is stored as log(K) for unconstrained optimization
            - Projection to valid range happens via clamping in K property
        """
        super().__init__()
        self.config = config or LorentzConfig()
        self.n = self.config.intrinsic_dim
        self.eps = self.config.eps

        # Store log(K) for unconstrained optimization
        val = math.log(self.config.initial_curvature)
        if self.config.learnable_curvature:
            self._log_K = nn.Parameter(torch.tensor(val))
        else:
            self.register_buffer("_log_K", torch.tensor(val))

    @property
    def K(self) -> torch.Tensor:
        """
        Curvature parameter K (sectional curvature is -1/K).

        Returns:
            Clamped curvature value ensuring manifold stability.

        Notes:
            - Space: Scalar parameter defining manifold geometry
            - Status: Exact value with bounds for numerical stability
        """
        return torch.exp(self._log_K).clamp(
            min=self.config.curvature_min, max=self.config.curvature_max
        )

    def get_curvature(self) -> torch.Tensor:
        """
        Get current curvature parameter K.

        Alias for the K property for compatibility.

        Returns:
            Clamped curvature value.
        """
        return self.K

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE GEOMETRIC OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def minkowski_inner(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Lorentz (Minkowski) inner product: <x,y>_L = -x₀y₀ + Σᵢxᵢyᵢ.

        Args:
            x: First point(s) on manifold [..., n+1].
            y: Second point(s) on manifold [..., n+1].

        Returns:
            Inner product value(s) [..., 1].

        Notes:
            - Space: Ambient Minkowski space R^{n,1}
            - Status: Exact closed-form
            - For points on H^n: <x,x>_L = -1/K
        """
        return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(
            dim=-1, keepdim=True
        )

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points onto the Lorentz hyperboloid.

        Given spatial coordinates x[..., 1:], computes x₀ such that
        <x,x>_L = -1/K, i.e., x₀ = sqrt(1/K + ||x_s||²).

        Args:
            x: Points in ambient space [..., n+1].

        Returns:
            Points projected onto hyperboloid [..., n+1].

        Notes:
            - Space: Projection from R^{n+1} onto H^n
            - Status: Exact closed-form projection
            - This is a "paranoid projection" when used for numerical correction
        """
        xs = x[..., 1:]
        x0 = torch.sqrt(
            1.0 / self.K + (xs**2).sum(dim=-1, keepdim=True) + self.eps
        )
        return torch.cat([x0, xs], dim=-1)

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate origin point(s) on the hyperboloid.

        The origin is (1/sqrt(K), 0, 0, ..., 0) in Lorentz coordinates.

        Args:
            batch_size: Number of origin points to generate.

        Returns:
            Origin points [batch_size, n+1].

        Notes:
            - Space: Point on manifold H^n
            - Status: Exact closed-form
            - Device synchronized with curvature parameter
        """
        device = self._log_K.device
        dtype = self._log_K.dtype
        o = torch.zeros(batch_size, self.n + 1, device=device, dtype=dtype)
        o[:, 0] = 1.0 / torch.sqrt(self.K)
        return o

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance between points on the hyperboloid.

        d_H(x,y) = (1/sqrt(K)) * acosh(-K * <x,y>_L)

        Args:
            x: First point(s) [..., n+1].
            y: Second point(s) [..., n+1].

        Returns:
            Geodesic distances [...].

        Notes:
            - Space: Geodesic distance on manifold H^n
            - Status: Exact closed-form
            - Always non-negative; d(x,x) = 0
        """
        inner = self.minkowski_inner(x, y).squeeze(-1)
        cosh_dist = (-self.K * inner).clamp(min=1.0 + self.eps)
        return safe_acosh(cosh_dist) / torch.sqrt(self.K)

    def distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise geodesic distance matrix.

        Args:
            x: Points [B, n+1].

        Returns:
            Distance matrix [B, B] where D[i,j] = d_H(x_i, x_j).

        Notes:
            - Space: Batch pairwise distances on H^n
            - Status: Exact closed-form (batch computation)
        """
        inner = self.minkowski_inner(
            x.unsqueeze(1), x.unsqueeze(0)
        ).squeeze(-1)
        cosh_dist = (-self.K * inner).clamp(min=1.0 + self.eps)
        return safe_acosh(cosh_dist) / torch.sqrt(self.K)

    def distance_matrix_points(
        self, x: torch.Tensor, y: torch.Tensor, pairwise: bool = True
    ) -> torch.Tensor:
        """
        Distance matrix between two sets of points.

        Args:
            x: First set of points [B1, n+1].
            y: Second set of points [B2, n+1].
            pairwise: If True, returns [B1, B2] matrix; else [B1] vector.

        Returns:
            Distance matrix [B1, B2] or vector [B1].

        Notes:
            - Space: Cross-set distances on H^n
            - Status: Exact closed-form
        """
        if pairwise:
            inner = self.minkowski_inner(
                x.unsqueeze(1), y.unsqueeze(0)
            ).squeeze(-1)
        else:
            inner = self.minkowski_inner(x, y).squeeze(-1)

        cosh_dist = (-self.K * inner).clamp(min=1.0 + self.eps)
        return safe_acosh(cosh_dist) / torch.sqrt(self.K)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPONENTIAL AND LOGARITHMIC MAPS
    # ═══════════════════════════════════════════════════════════════════════════

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: moves from x in direction v (tangent vector).

        exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * (v / ||v||_L)

        where ||v||_L = sqrt(<v,v>_L) is the Lorentzian norm of v.

        Args:
            x: Base point(s) on manifold [..., n+1].
            v: Tangent vector(s) at x [..., n+1].

        Returns:
            Endpoint(s) of geodesic [..., n+1].

        Notes:
            - Space: Map from tangent bundle TH^n to manifold H^n
            - Status: Exact closed-form exponential map
            - v must satisfy <x,v>_L = 0 (tangent space constraint)
        """
        v_norm_sq = self.minkowski_inner(v, v).clamp(min=self.eps)
        v_norm = torch.sqrt(v_norm_sq)

        # Scale by sqrt(K) for curvature
        scaled_norm = v_norm * torch.sqrt(self.K)

        result = (
            torch.cosh(scaled_norm) * x
            + torch.sinh(scaled_norm) * v / (v_norm + self.eps)
        )

        # Paranoid projection for numerical stability
        return self.proj(result)

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from the origin.

        Specialized version of exp_map when base point is the origin.

        Args:
            v: Tangent vectors at origin [..., n+1].

        Returns:
            Points on manifold [..., n+1].

        Notes:
            - Space: Map from T_o H^n to H^n
            - Status: Exact closed-form (origin specialization)
        """
        origin = self.origin(v.shape[0]).to(v.device, v.dtype)
        return self.exp_map(origin, v)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: inverse of exponential map.

        log_x(y) returns the tangent vector v at x such that exp_x(v) = y.

        log_x(y) = (d_H(x,y) / sinh(sqrt(K)*d_H(x,y))) * (y - cosh(sqrt(K)*d_H(x,y))*x)

        Args:
            x: Base point(s) on manifold [..., n+1].
            y: Target point(s) on manifold [..., n+1].

        Returns:
            Tangent vector(s) at x [..., n+1].

        Notes:
            - Space: Map from H^n × H^n to tangent bundle TH^n
            - Status: Exact closed-form logarithmic map
            - Singular only when x = y (returns zero vector)
        """
        d = self.dist(x, y).unsqueeze(-1)
        scaled_d = d * torch.sqrt(self.K)

        # Avoid division by zero when d ≈ 0
        sinh_d = torch.sinh(scaled_d).clamp(min=self.eps)

        v = (d / sinh_d) * (y - torch.cosh(scaled_d) * x)

        # Project to ensure tangent space constraint
        return self.proj_tangent(x, v)

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map to the origin.

        Args:
            y: Points on manifold [..., n+1].

        Returns:
            Tangent vectors at origin [..., n+1].

        Notes:
            - Space: Map from H^n to T_o H^n
            - Status: Exact closed-form (origin specialization)
        """
        origin = self.origin(y.shape[0]).to(y.device, y.dtype)
        return self.log_map(origin, y)

    def proj_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector onto tangent space at x.

        Ensures <x,v>_L = 0 by subtracting the component along x.

        Args:
            x: Point(s) on manifold [..., n+1].
            v: Vector(s) to project [..., n+1].

        Returns:
            Projected tangent vector(s) [..., n+1].

        Notes:
            - Space: Projection onto T_x H^n
            - Status: Exact projection (paranoid correction)
            - This is a numerical correction, not a theoretical modification
        """
        inner = self.minkowski_inner(x, v)
        return v + self.K * inner * x

    # ═══════════════════════════════════════════════════════════════════════════
    # RIEMANNIAN OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def riemannian_grad(
        self, x: torch.Tensor, grad_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert Euclidean gradient to Riemannian gradient.

        The Riemannian gradient is the projection of the Euclidean gradient
        onto the tangent space, scaled by the inverse metric.

        g_R = g_E + K * <x, g_E>_L * x

        Args:
            x: Point(s) on manifold [..., n+1].
            grad_e: Euclidean gradient(s) [..., n+1].

        Returns:
            Riemannian gradient(s) [..., n+1].

        Notes:
            - Space: Conversion from ambient gradient to tangent gradient
            - Status: Exact Riemannian gradient via metric conversion
        """
        inner = self.minkowski_inner(x, grad_e)
        return grad_e + self.K * inner * x

    # ═══════════════════════════════════════════════════════════════════════════
    # PARALLEL TRANSPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def parallel_transport(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport of tangent vector v from x to y.

        Uses closed-form parallel transport along geodesics.

        Args:
            x: Source point(s) [..., n+1].
            y: Target point(s) [..., n+1].
            v: Tangent vector(s) at x [..., n+1].

        Returns:
            Transported vector(s) at y [..., n+1].

        Notes:
            - Space: Parallel transport along geodesic γ: x → y
            - Status: Exact closed-form parallel transport
        """
        log_xy = self.log_map(x, y)
        norm = torch.sqrt(self.minkowski_inner(log_xy, log_xy).clamp(min=self.eps))

        # Transport formula
        inner_v_log = self.minkowski_inner(v, log_xy)
        inner_log_x = self.minkowski_inner(log_xy, x)

        result = v - (inner_v_log / (norm**2 + self.eps)) * (
            log_xy + self.log_map(y, x)
        )

        return self.proj_tangent(y, result)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMILARITY METRICS
    # ═══════════════════════════════════════════════════════════════════════════

    def lorentz_similarity(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Lorentz-based similarity: 1 / (-K * <x,y>_L).

        Returns similarity in (0, 1], where 1 indicates identical points.

        Args:
            x: First point(s) [..., n+1].
            y: Second point(s) [..., n+1].

        Returns:
            Similarity values [...].

        Notes:
            - Space: Similarity metric on H^n
            - Status: Exact closed-form (derived from inner product)
        """
        inner = self.minkowski_inner(x, y).squeeze(-1)
        cosh_dist = (-self.K * inner).clamp(min=1.0 + self.eps)
        return 1.0 / cosh_dist

    def geodesic_similarity(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Geodesic similarity: exp(-d_H(x,y)).

        METRIC-CONSISTENT with training that uses -D as logits.

        Args:
            x: First point(s) [..., n+1].
            y: Second point(s) [..., n+1].

        Returns:
            Similarity values in (0, 1] [...].

        Notes:
            - Space: Exponential decay of geodesic distance
            - Status: Exact closed-form
            - Use this for evaluation to maintain train/eval consistency
        """
        d = self.dist(x, y)
        return torch.exp(-d)

    # ═══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def manifold_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean deviation from manifold constraint.

        For points on H^n: <x,x>_L = -1/K. This measures ||<x,x>_L + 1/K||.

        Args:
            x: Points to check [..., n+1].

        Returns:
            Mean absolute violation (scalar).

        Notes:
            - Space: Diagnostic for manifold membership
            - Status: Exact constraint violation measure
        """
        inner = self.minkowski_inner(x, x).squeeze(-1)
        target = -1.0 / self.K
        return torch.abs(inner - target).mean()

    def lorentz_radius(self, x: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance from point x to hyperboloid origin.

        Essential for monitoring radial distribution and detecting collapse.

        Args:
            x: Points on manifold [..., n+1].

        Returns:
            Radial distances [...].

        Notes:
            - Space: Distance to origin on H^n
            - Status: Exact closed-form
        """
        x0 = x[..., 0]
        arg = (x0 * torch.sqrt(self.K)).clamp(min=1.0 + self.eps)
        return torch.acosh(arg) / torch.sqrt(self.K)

    # =========================================================================
    # COMPATIBILITY ALIASES (for notebook/module interoperability)
    # =========================================================================

    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for proj() - notebook compatibility.
        
        See proj() for full documentation.
        """
        return self.proj(x)

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Alias for dist() - notebook compatibility.
        
        See dist() for full documentation.
        """
        return self.dist(x, y)

    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Minkowski norm of a vector.
        
        For timelike vectors (on tangent space), this gives the
        "length" under Minkowski metric.
        
        Args:
            x: Vector in Minkowski space [..., n+1].
            
        Returns:
            Minkowski norm [...].
            
        Notes:
            - For tangent vectors at a point on H^n, this is the
              Riemannian norm induced by the Minkowski metric.
            - Status: Exact closed-form.
        """
        inner = self.minkowski_inner(x, x)
        # For spacelike/tangent vectors, inner is positive
        return safe_sqrt(torch.abs(inner))
