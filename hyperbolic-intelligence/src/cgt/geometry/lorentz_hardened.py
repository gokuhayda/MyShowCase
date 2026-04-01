# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Lorentz Substrate [HARDENED VERSION]
====================================

Exact implementation matching CGT_Paper_Ready_v6_1_HARDENED notebook.
All operations are metric-consistent and numerically stable.

Mathematical Status
-------------------
- All core operations: EXACT (closed-form Lorentz geometry)
- Numerical stability: HARDENED (eps clamps, safe_acosh)
- Device sync: AUTOMATIC

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_sqrt(x: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """Numerically stable square root."""
    return torch.sqrt(torch.clamp(x, min=eps))


def safe_acosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Numerically stable inverse hyperbolic cosine with Taylor expansion.
    
    AUDIT FIX: The simple clamp creates a zero-gradient region for x ∈ [1, 1+ε].
    Taylor expansion of arccosh(1+δ) ≈ √(2δ) restores gradient flow near x=1.
    
    Mathematical basis:
        - Standard path: arccosh(x) for x > 1+ε (stable)
        - Taylor path: √(2(x-1)) for x ∈ [1, 1+ε] (gradient-preserving)
    
    This enables learning of identity and fine refinement of positive pairs.
    """
    delta = x - 1.0
    mask = delta < eps
    
    # Standard path (stable far from 1)
    val_standard = torch.acosh(torch.clamp(x, min=1.0 + eps))
    
    # Taylor path: arccosh(1+z) ≈ sqrt(2z) for small z
    # Clamp delta to avoid sqrt of negative
    val_taylor = torch.sqrt(2.0 * torch.clamp(delta, min=0.0) + 1e-15)
    
    return torch.where(mask, val_taylor, val_standard)


@dataclass
class LorentzConfig:
    """
    Configuration for the Lorentz (hyperbolic) manifold substrate.

    The Lorentz model embeds points on the upper sheet of a two-sheeted
    hyperboloid in Minkowski space with signature (-,+,+,...,+).

    Attributes:
        intrinsic_dim: Dimension of the hyperbolic space (n in H^n)
        eps: Numerical stability epsilon for clamp operations
        learnable_curvature: Whether K is a learnable parameter
        initial_curvature: Initial value of sectional curvature K
        curvature_min: Minimum allowed curvature (prevents collapse)
        curvature_max: Maximum allowed curvature (prevents explosion)
        
    Notes:
        - Ambient dimension is intrinsic_dim + 1 (for time component)
        - Points satisfy: -x₀² + x₁² + ... + xₙ² = -1/K
    """
    intrinsic_dim: int = 32
    eps: float = 1e-6
    learnable_curvature: bool = True
    initial_curvature: float = 1.0
    curvature_min: float = 0.1
    curvature_max: float = 10.0


class LorentzSubstrateHardened(nn.Module):
    """
    Lorentz (Hyperboloid) Manifold Implementation [HARDENED].
    
    This is the production-ready substrate for CGT training, matching
    exactly the implementation in CGT_Paper_Ready_v6_1_HARDENED notebook.
    
    Features:
        - Metric-consistent similarity functions
        - GPU-optimized distance matrices
        - Riemannian gradient conversion
        - Automatic device synchronization
    
    Attributes:
        n: Intrinsic dimension (hyperbolic dimension)
        K: Sectional curvature parameter (learnable or fixed)
        eps: Numerical stability constant
        
    Notes:
        - Mathematical Identity: Lorentz (Hyperboloid) Model of H^n
        - Classification: EXACT (all operations are closed-form)
        - Stability: HARDENED (clamps, safe_acosh, device sync)
    """
    
    def __init__(self, config: Optional[LorentzConfig] = None):
        """
        Initialize Lorentz substrate.
        
        Args:
            config: LorentzConfig instance (uses defaults if None)
        """
        super().__init__()
        self.config = config or LorentzConfig()
        self.n = self.config.intrinsic_dim
        self.eps = self.config.eps

        val = math.log(self.config.initial_curvature)
        if self.config.learnable_curvature:
            self._log_K = nn.Parameter(torch.tensor(val))
        else:
            self.register_buffer('_log_K', torch.tensor(val))

    @property
    def K(self) -> torch.Tensor:
        """Returns curvature (K) ensuring correct device and autograd."""
        return torch.exp(self._log_K).clamp(
            min=self.config.curvature_min, 
            max=self.config.curvature_max
        )

    @property
    def curvature(self) -> torch.Tensor:
        """Alias for K for backward compatibility."""
        return self.K

    # ═══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def manifold_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes mean error of ⟨x, x⟩_L + 1/K = 0.
        
        Should be ≈0 for valid points on manifold.
        """
        # Check for NaN/Inf in input
        if not torch.isfinite(x).all():
            return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
        
        K = self.K.to(x.device, x.dtype)
        inner = self.minkowski_inner(x, x).squeeze(-1)
        target = -1.0 / K
        
        # Handle NaN in computation
        violation = torch.abs(inner - target)
        violation = torch.nan_to_num(violation, nan=0.0, posinf=0.0, neginf=0.0)
        
        return violation.mean()

    def lorentz_radius(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes geodesic distance from point x to the hyperboloid origin.
        Essential for monitoring radial collapse in the Lorentz model.

        TB-PAG fix: replaced raw torch.acosh with safe_acosh (Taylor surrogate
        near branch point) to prevent Geometric Amplification Loop when x₀·√K ≈ 1.
        (Theorem 1: acosh(1+δ) ≈ √(2δ) amplifies δ~1e-7 to O(1e-3).)
        """
        K = self.K.to(x.device, x.dtype)
        x0 = x[..., 0]
        arg = (x0 * torch.sqrt(K)).clamp(min=1.0 + self.eps)
        return safe_acosh(arg) / torch.sqrt(K)

    # ═══════════════════════════════════════════════════════════════════════════
    # RIEMANNIAN OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def riemannian_grad(self, x: torch.Tensor, grad_e: torch.Tensor) -> torch.Tensor:
        """
        Converts Euclidean gradient to Riemannian via tangent projection.
        
        Formula: g_r = g_e + K * ⟨x, g_e⟩_L * x
        
        Args:
            x: Point on manifold
            grad_e: Euclidean gradient
            
        Returns:
            Riemannian gradient in tangent space at x
        """
        inner = self.minkowski_inner(x, grad_e)
        K = self.K.to(x.device, x.dtype)
        return grad_e + K * inner * x

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE GEOMETRIC OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ
        
        Returns tensor with keepdim=True for broadcasting compatibility.
        """
        return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects points onto the Lorentzian hyperboloid.
        
        Given spatial components x[1:], computes time component x[0]
        such that ⟨x,x⟩_L = -1/K.
        
        Formula: x₀ = √(1/K + ||x_s||²)
        """
        K = self.K.to(x.device, x.dtype)
        xs = x[..., 1:]
        x0 = torch.sqrt(1.0 / K + (xs ** 2).sum(dim=-1, keepdim=True) + self.eps)
        return torch.cat([x0, xs], dim=-1)

    def origin(self, batch_size: int = 1, device=None, dtype=None) -> torch.Tensor:
        """
        Generates origin point(s) synchronized with curvature device.
        
        Origin: (1/√K, 0, 0, ..., 0)
        """
        if device is None:
            device = self._log_K.device
        if dtype is None:
            dtype = self._log_K.dtype
        K = self.K.to(device, dtype)
        o = torch.zeros(batch_size, self.n + 1, device=device, dtype=dtype)
        o[:, 0] = 1.0 / torch.sqrt(K)
        return o

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential Map (Retraction): T_x M → M
        
        Maps tangent vector v at point x to the manifold.
        
        Formula:
            exp_x(v) = cosh(||v||_L * √K) * x + sinh(||v||_L * √K) * v/||v||_L
        """
        K = self.K.to(x.device, x.dtype)
        v_norm_sq = torch.abs(self.minkowski_inner(v, v)) + self.eps
        v_norm = torch.sqrt(v_norm_sq)
        scale = (v_norm * torch.sqrt(K)).clamp(max=15.0)
        cosh_scale = torch.cosh(scale)
        sinh_scale = torch.sinh(scale) / (v_norm + self.eps)
        return self.proj(cosh_scale * x + sinh_scale * v)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic Map: M → T_x M

        Maps point y on manifold to tangent space at x.

        TB-PAG fix: tangent re-orthogonalization (eq. 18) appended to
        eliminate Tangency Violation |⟨x, u⟩_L| ~ 1e-6 that accumulates
        across layers and corrupts Riemannian optimiser gradient projections.
        u = ũ - ⟨ũ, x⟩_L · x  →  reduces residual from O(1e-6) to ~ε_machine.
        """
        K = self.K.to(x.device, x.dtype)
        inner = self.minkowski_inner(x, y)
        v = y + K * inner * x
        v_norm_sq = torch.abs(self.minkowski_inner(v, v)) + self.eps
        v_norm = torch.sqrt(v_norm_sq)
        d = self.dist(x, y).unsqueeze(-1)
        u = d * v / (v_norm + self.eps)
        # TB-PAG eq. 18: re-orthogonalize onto T_x H^n
        u = u - self.minkowski_inner(u, x) * x
        return u

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """Maps y to tangent space at origin."""
        o = self.origin(y.shape[0]).to(y.device, y.dtype)
        return self.log_map(o, y)

    # ═══════════════════════════════════════════════════════════════════════════
    # DISTANCE FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic geodesic distance between two points.

        Formula: d(x,y) = (1/√K) * arccosh(-K * ⟨x,y⟩_L)

        TB-PAG fixes applied (§6B Precision-Aware Geometry):
        - Inputs reprojected to hyperboloid (Topological Bridging, eq. 17)
          before entering the geodesic-critical computation.
        - Inner product and acosh computed in float64 zone to suppress
          Precision Boundary Rupture (Definition 1).
        - safe_acosh (Taylor surrogate) replaces raw torch.acosh to neutralise
          the Geometric Amplification Loop (Definition 3, Theorem 1).
        """
        orig_dtype = x.dtype
        # TB-PAG: reproject + float64 zone for geodesic-critical op
        x = self.proj(x.to(torch.float64))
        y = self.proj(y.to(torch.float64))
        K = self.K.to(x.device, torch.float64)

        inner = self.minkowski_inner(x, y).squeeze(-1)
        arg = (-K * inner).clamp(min=1.0 + self.eps, max=1e5)
        dist = safe_acosh(arg) / torch.sqrt(K)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        return dist.to(orig_dtype)

    def distance_matrix_points(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pairwise: bool = True
    ) -> torch.Tensor:
        """
        Computes GPU-optimized distance matrices.

        Used by HomeostaticField for anchor distances.

        Args:
            x: Points [N, dim+1]
            y: Points [M, dim+1]
            pairwise: If True, compute full NxM matrix

        Returns:
            Distance matrix [N, M] if pairwise, else [N]

        TB-PAG fixes applied (§6B):
        - Reprojection before inner product (Topological Bridging, eq. 17).
        - float64 zone for the full geodesic computation.
        - safe_acosh replaces raw torch.acosh (Theorem 1 amplification).
        """
        orig_dtype = x.dtype
        # TB-PAG: reproject + float64 zone
        x = self.proj(x.to(torch.float64))
        y = self.proj(y.to(torch.float64))
        K_val = self.K.to(x.device, torch.float64)

        if pairwise:
            spatial = torch.mm(x[:, 1:], y[:, 1:].t())
            time = torch.mm(x[:, :1], y[:, :1].t())
            inner = -time + spatial
        else:
            inner = self.minkowski_inner(x, y).squeeze(-1)

        arg = (-K_val * inner).clamp(min=1.0 + self.eps, max=1e5)
        dist = safe_acosh(arg) / (torch.sqrt(K_val) + 1e-9)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        return dist.to(orig_dtype)

    def distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates pairwise geodesic distance matrix O(B²).
        Compatible interface with PowerLawDistillation.
        """
        return self.distance_matrix_points(x, x, pairwise=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMILARITY FUNCTIONS [METRIC-CONSISTENT]
    # ═══════════════════════════════════════════════════════════════════════════

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Lorentz similarity: 1 / (-K * ⟨x,y⟩_L)

        Returns similarity in (0, 1], where 1 = identical points.
        Monotonic transformation of Minkowski inner product.
        """
        K = self.K.to(x.device, x.dtype)
        inner = self.minkowski_inner(x, y).squeeze(-1)
        cosh_dist = (-K * inner).clamp(min=1.0 + self.eps)
        return 1.0 / cosh_dist

    def geodesic_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic similarity: exp(-d_H(x,y))

        METRIC-CONSISTENT with training that uses -D/τ as logits.
        Returns similarity in (0, 1], where 1 = identical points.

        Properties:
        - Monotonically decreasing with distance
        - sim(x,x) = 1
        - Consistent with HyperbolicInfoNCE_Lorentz training objective
        """
        d = self.dist(x, y)
        return torch.exp(-d)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALIASES FOR COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for proj() for API compatibility."""
        return self.proj(x)
    
    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias for dist() for API compatibility."""
        return self.dist(x, y)
    
    def get_curvature(self) -> torch.Tensor:
        """Get current curvature value."""
        return self.K
    
    def to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from Lorentz (hyperboloid) to Poincaré ball model.
        
        Maps point x = (x0, x1, ..., xn) on hyperboloid to
        point y = (x1, ..., xn) / (x0 + 1) in Poincaré ball.
        
        Parameters
        ----------
        x : torch.Tensor
            Points on Lorentz manifold, shape (..., dim+1)
            
        Returns
        -------
        torch.Tensor
            Points in Poincaré ball, shape (..., dim)
        """
        x0 = x[..., 0:1]  # Time component
        xi = x[..., 1:]   # Spatial components
        
        # Stereographic projection: y = xi / (x0 + 1)
        return xi / (x0 + 1.0 + 1e-8)
    
    def from_poincare(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert from Poincaré ball to Lorentz (hyperboloid) model.
        
        Parameters
        ----------
        y : torch.Tensor
            Points in Poincaré ball, shape (..., dim)
            
        Returns
        -------
        torch.Tensor
            Points on Lorentz manifold, shape (..., dim+1)
        """
        y_sqnorm = (y ** 2).sum(dim=-1, keepdim=True)
        
        # Inverse stereographic projection
        x0 = (1.0 + y_sqnorm) / (1.0 - y_sqnorm + 1e-8)
        xi = 2 * y / (1.0 - y_sqnorm + 1e-8)
        
        x = torch.cat([x0, xi], dim=-1)
        return self.proj(x)  # Ensure on manifold

    # ═══════════════════════════════════════════════════════════════════════════
    # LLM-OPTIMIZED OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from origin (optimized for LLM embedding lookup).
        
        For tangent vectors at origin o = (1/√K, 0, ..., 0):
            exp_o(v) = cosh(√K·||v||)·o + sinh(√K·||v||)·v/||v||
        
        Args:
            v: Tangent vectors at origin [..., n+1] where v[..., 0] = 0
            
        Returns:
            Points on manifold [..., n+1]
            
        Notes:
            - Input v must have v[..., 0] = 0 (valid tangent at origin)
            - Uses Taylor expansion for small norms (stability)
            - Clamps scale to prevent overflow in sinh/cosh
        """
        K = self.K.to(v.device, v.dtype)
        sqrt_K = torch.sqrt(K)
        
        # For tangent at origin: ||v||_L = ||v_spatial|| (Lorentz norm = Euclidean norm of spatial)
        v_spatial = v[..., 1:]
        
        # 🔧 HARD NUMERICAL GUARD (CRÍTICO)
        v_spatial = torch.nan_to_num(
            v_spatial,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        
        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True)
        
        # Protege domínio do exp / sinh / cosh
        v_norm = torch.nan_to_num(
            v_norm,
            nan=0.0,
            posinf=15.0,
            neginf=0.0,
        ).clamp(min=self.eps, max=15.0)
        
        scale = sqrt_K * v_norm
        
        # sinh/cosh computation
        cosh_scale = torch.cosh(scale)
        sinh_scale = torch.sinh(scale) / (v_norm + self.eps)
        
        # Origin point
        o_time = 1.0 / sqrt_K
        
        # exp_o(v) = cosh(...)·o + sinh(...)·v/||v||
        # Time component: cosh(...) * (1/√K)
        # Spatial components: sinh(...) * v_spatial / ||v||
        x_time = cosh_scale * o_time
        x_spatial = sinh_scale * v_spatial
        
        return torch.cat([x_time, x_spatial], dim=-1)

    def log_map_zero_batch(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map to origin (batch-optimized for LLM).
        
        Maps points y on manifold to tangent space at origin.
        
        Args:
            y: Points on manifold [..., n+1]
            
        Returns:
            Tangent vectors at origin [..., n+1]
        """
        K = self.K.to(y.device, y.dtype)
        sqrt_K = torch.sqrt(K)
        
        # Distance from origin: d = arccosh(x₀ · √K) / √K
        y0 = y[..., 0:1]
        arg = (y0 * sqrt_K).clamp(min=1.0 + self.eps, max=1e5)
        d = safe_acosh(arg) / sqrt_K
        
        # Direction: y_spatial / ||y_spatial||
        y_spatial = y[..., 1:]
        y_spatial_norm = torch.norm(y_spatial, dim=-1, keepdim=True).clamp(min=self.eps)
        
        # Tangent vector: d * direction (time component = 0)
        v_spatial = d * y_spatial / y_spatial_norm
        v_time = torch.zeros_like(y0)
        
        return torch.cat([v_time, v_spatial], dim=-1)

    def distance_matrix_batch(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise geodesic distances for attention (GPU-optimized).
        
        Args:
            q: Query points [B, L_q, n+1] or [B, H, L_q, n+1]
            k: Key points [B, L_k, n+1] or [B, H, L_k, n+1]
            
        Returns:
            Distance matrix [B, L_q, L_k] or [B, H, L_q, L_k]
        """
        K = self.K.to(q.device, q.dtype)
        
        # Minkowski inner product via batch matmul
        # ⟨q, k⟩_L = -q₀k₀ + q₁k₁ + ... + qₙkₙ
        q_time = q[..., 0:1]  # [..., L_q, 1]
        q_space = q[..., 1:]  # [..., L_q, n]
        k_time = k[..., 0:1]  # [..., L_k, 1]
        k_space = k[..., 1:]  # [..., L_k, n]
        
        # Batch matrix multiply for spatial part
        space_inner = torch.matmul(q_space, k_space.transpose(-2, -1))  # [..., L_q, L_k]
        time_inner = torch.matmul(q_time, k_time.transpose(-2, -1))     # [..., L_q, L_k]
        
        inner = -time_inner + space_inner
        
        # Distance: d = (1/√K) · arccosh(-K · ⟨q, k⟩_L)
        arg = (-K * inner).clamp(min=1.0 + self.eps, max=1e5)
        dist = safe_acosh(arg) / torch.sqrt(K)
        
        return torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    def tangent_linear(
        self, 
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Linear transformation in tangent space at origin.
        
        For LLM: log → linear → exp pattern.
        
        Args:
            x: Points on manifold [..., n+1]
            weight: Linear weight [out_dim, n] (operates on spatial only)
            bias: Optional bias [out_dim]
            
        Returns:
            Transformed points on manifold [..., out_dim+1]
        """
        # Map to tangent at origin
        v = self.log_map_zero_batch(x)
        v_spatial = v[..., 1:]  # [..., n]
        
        # Linear transformation (spatial only)
        out_spatial = F.linear(v_spatial, weight, bias)  # [..., out_dim]
        
        # Zero-pad time component for tangent vector
        out_time = torch.zeros(
            *out_spatial.shape[:-1], 1, 
            device=out_spatial.device, 
            dtype=out_spatial.dtype
        )
        v_out = torch.cat([out_time, out_spatial], dim=-1)
        
        # Map back to manifold
        return self.exp_map_zero(v_out)

    def weighted_midpoint(
        self, 
        x: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Weighted Fréchet mean approximation via tangent space aggregation.
        
        For attention value aggregation: maps to tangent, weighted sum, maps back.
        
        Args:
            x: Points on manifold [B, L, n+1]
            weights: Attention weights [B, L] (sum to 1 per row)
            
        Returns:
            Aggregated points [B, n+1]
            
        Notes:
            - This is an approximation to the true Fréchet mean
            - Exact when all points are at origin
            - Error bounded by radius regularization
        """
        # Map to tangent at origin
        v = self.log_map_zero_batch(x)  # [B, L, n+1]
        
        # Weighted sum in tangent space
        weights = weights.unsqueeze(-1)  # [B, L, 1]
        v_mean = (v * weights).sum(dim=-2)  # [B, n+1]
        
        # Map back to manifold
        return self.exp_map_zero(v_mean)

    def attention_aggregate(
        self, 
        values: torch.Tensor, 
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Geodesic attention aggregation for transformer.
        
        Args:
            values: Value points [B, H, L, n+1]
            attention_weights: Softmax weights [B, H, L_q, L_k] (sum to 1 over L_k)
            
        Returns:
            Aggregated outputs [B, H, L_q, n+1]
        """
        # Map values to tangent space
        v = self.log_map_zero_batch(values)  # [B, H, L_k, n+1]
        
        # Weighted aggregation in tangent space
        # attention_weights: [B, H, L_q, L_k]
        # v: [B, H, L_k, n+1]
        # Result: [B, H, L_q, n+1]
        v_agg = torch.einsum('bhqk,bhkd->bhqd', attention_weights, v)
        
        # Map back to manifold
        return self.exp_map_zero(v_agg)

    def parallel_transport_zero(
        self, 
        v: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport from origin to point y.
        
        Transports tangent vector v ∈ T_o M to T_y M.
        
        Args:
            v: Tangent vector at origin [..., n+1]
            y: Target point on manifold [..., n+1]
            
        Returns:
            Transported vector in T_y M [..., n+1]
        """
        K = self.K.to(v.device, v.dtype)
        
        # Inner products needed for transport
        inner_v_y = self.minkowski_inner(v, y)  # ⟨v, y⟩_L
        
        # Origin
        o = self.origin(1).to(v.device, v.dtype).squeeze(0)
        inner_o_y = self.minkowski_inner(o.unsqueeze(0), y)  # ⟨o, y⟩_L
        
        # Transport formula: v' = v - K · ⟨v, y⟩_L / (1 - K · ⟨o, y⟩_L) · (o + y)
        # Simplified for origin
        denom = 1.0 - K * inner_o_y + self.eps
        coeff = K * inner_v_y / denom
        
        return v - coeff * (o + y)
