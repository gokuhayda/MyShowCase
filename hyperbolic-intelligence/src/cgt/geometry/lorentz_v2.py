"""
cgt_v2/geometry/lorentz_v2.py
==============================
Fully isolated Lorentz (Hyperboloid) manifold substrate for v2.

Zero imports from the legacy cgt.* namespace.

Mathematical contract
---------------------
Every point x on H^n satisfies the Minkowski constraint:
    <x, x>_L  =  -x₀² + x₁² + … + xₙ²  =  -1/K

where K > 0 is the sectional curvature and the ambient dim is n+1.

All geometry ops (exp_map, log_map, proj, dist) enforce this constraint.
All ops that cross the Euclidean ↔ hyperbolic boundary use float64
internally and cast back to the caller's dtype only after proj().

Precision policy (TB-PAG)
--------------------------
- minkowski_inner  → computed in float64, returned in caller dtype
- dist / log_map   → float64 throughout, returned in float64
- exp_map          → float64 for cosh/sinh, proj in float64, then cast
- log_map_zero     → delegates to log_map (float64)
- proj             → works in any dtype (caller responsible for casting)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Numerically stable primitives
# ─────────────────────────────────────────────────────────────────────────────

def safe_sqrt_v2(x: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """Numerically stable sqrt: clamp before sqrt to prevent NaN gradient."""
    return torch.sqrt(x.clamp(min=eps))


def safe_acosh_v2(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Numerically stable arccosh with Taylor expansion near x = 1.

    Standard acosh has zero gradient at x = 1 (clamp-based implementations).
    Taylor expansion arccosh(1+δ) ≈ √(2δ) restores gradient flow.

        - x < 1+eps  →  Taylor path: √(2(x-1))   [gradient-preserving]
        - x ≥ 1+eps  →  Standard:   acosh(x)      [exact]
    """
    delta = x - 1.0
    mask = delta < eps

    val_std = torch.acosh(x.clamp(min=1.0 + eps))
    val_taylor = torch.sqrt((2.0 * delta.clamp(min=0.0) + 1e-15))

    return torch.where(mask, val_taylor, val_std)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LorentzConfigV2:
    """
    Configuration for the v2 Lorentz substrate.

    Config Precedence Rule (MANDATORY — applied in __post_init__)
    -------------------------------------------------------------
    For every field that has a canonical name and a legacy alias:

        if canonical_value != default:
            use canonical_value
        elif alias_value is not None:
            use alias_value
        else:
            use default

    Fields
    ------
    intrinsic_dim       : Hyperbolic dimension n  (H^n has ambient dim n+1)
    eps                 : Global numerical stability floor
    learnable_curvature : Whether K is a learned nn.Parameter
    initial_curvature   : Starting value of K
    curvature_min       : Hard lower clamp on K
    curvature_max       : Hard upper clamp on K

    Legacy aliases (accepted but not canonical)
    -------------------------------------------
    dim                 : alias for intrinsic_dim
    curvature           : alias for initial_curvature
    """

    # ── canonical fields ──────────────────────────────────────────────────
    intrinsic_dim: int       = 32
    eps: float               = 1e-6
    learnable_curvature: bool = True
    initial_curvature: float = 1.0
    curvature_min: float     = 0.1
    curvature_max: float     = 10.0

    # ── legacy aliases (set to sentinel None so precedence rule can fire) ─
    dim: Optional[int]       = None   # alias for intrinsic_dim
    curvature: Optional[float] = None  # alias for initial_curvature

    def __post_init__(self) -> None:
        """Apply config precedence rule to all aliased fields."""
        _DEFAULT_INTRINSIC_DIM = 32
        _DEFAULT_INITIAL_CURVATURE = 1.0

        # intrinsic_dim  ←  canonical > alias > default
        if self.intrinsic_dim != _DEFAULT_INTRINSIC_DIM:
            pass  # canonical wins, nothing to do
        elif self.dim is not None:
            self.intrinsic_dim = self.dim
        # else: keep default

        # initial_curvature  ←  canonical > alias > default
        if self.initial_curvature != _DEFAULT_INITIAL_CURVATURE:
            pass  # canonical wins
        elif self.curvature is not None:
            self.initial_curvature = self.curvature
        # else: keep default


# ─────────────────────────────────────────────────────────────────────────────
# Substrate
# ─────────────────────────────────────────────────────────────────────────────

class LorentzSubstrateV2(nn.Module):
    """
    Lorentz Hyperboloid Manifold Substrate — v2 (geometry-correct, isolated).

    Public API
    ----------
    proj(x)                 → enforce Minkowski constraint
    origin(B, device, dtype)→ canonical origin point(s)
    exp_map(x, v)           → exponential map at x
    exp_map_zero(v)         → exp_map at origin (fast path)
    log_map(x, y)           → logarithmic map at x (returns float64)
    log_map_zero(y)         → log_map at origin (returns float64)
    dist(x, y)              → geodesic distance (float64 internally)
    minkowski_inner(x, y)   → Lorentz inner product (float64 internally)
    manifold_violation(x)   → mean |<x,x>_L + 1/K|  (diagnostic)

    Shape convention
    ----------------
    All manifold points: [..., n+1]  (time-first: [x₀, x₁, …, xₙ])
    Tangent vectors:     [..., n+1]  (v₀ = 0 at origin)
    """

    def __init__(self, config: Optional[LorentzConfigV2] = None) -> None:
        super().__init__()
        self.config = config or LorentzConfigV2()
        self.n = self.config.intrinsic_dim   # intrinsic (spatial) dimension
        self.eps = self.config.eps

        log_k_init = math.log(self.config.initial_curvature)
        if self.config.learnable_curvature:
            self._log_K: nn.Parameter | torch.Tensor = nn.Parameter(
                torch.tensor(log_k_init, dtype=torch.float64)
            )
        else:
            self.register_buffer(
                "_log_K", torch.tensor(log_k_init, dtype=torch.float64)
            )

    # ── Curvature property ────────────────────────────────────────────────

    @property
    def K(self) -> torch.Tensor:
        """Sectional curvature K > 0, clamped to [min, max]."""
        return torch.exp(self._log_K).clamp(
            min=self.config.curvature_min,
            max=self.config.curvature_max,
        )

    # ── Core geometry ─────────────────────────────────────────────────────

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Lorentz inner product: <x,y>_L = -x₀y₀ + Σᵢ xᵢyᵢ

        Computed in float64 to suppress catastrophic cancellation.
        Returns tensor in caller's original dtype with keepdim=True semantics
        (last dim is 1).
        """
        orig_dtype = x.dtype
        x64, y64 = x.double(), y.double()
        result = (
            -x64[..., :1] * y64[..., :1]
            + (x64[..., 1:] * y64[..., 1:]).sum(dim=-1, keepdim=True)
        )
        return result.to(orig_dtype)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto hyperboloid: given spatial x[1:], recompute x[0].

        x₀ = √(1/K + ‖x_spatial‖²)

        This enforces <x,x>_L = -1/K exactly (up to float precision).

        Always compute in float64, cast back to caller dtype at end.
        Reason: in float32 with ||x_spatial|| ~ 10-100 (common after exp_map),
        the sum of 128 squared terms has catastrophic cancellation that leaves
        the constraint error at ~1e-4 to 1e-2.
        Computing in float64 reduces this to ~1e-12, then the cast to float32
        only introduces a final rounding error of ~1e-7 (float32 ULP at x0~10).
        """
        orig_dtype = x.dtype
        x64 = x.double()
        K64 = self.K.double().to(x.device)
        xs64 = x64[..., 1:]
        x0_64 = torch.sqrt(
            (1.0 / K64 + (xs64 ** 2).sum(dim=-1, keepdim=True)).clamp(min=1e-15)
        )
        result64 = torch.cat([x0_64, xs64], dim=-1)
        return result64.to(orig_dtype)

    def origin(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return origin point(s) on H^n: o = (1/√K, 0, …, 0)   shape [B, n+1]
        """
        if device is None:
            device = self._log_K.device
        if dtype is None:
            dtype = torch.float32
        K = self.K.to(device, dtype)
        o = torch.zeros(batch_size, self.n + 1, device=device, dtype=dtype)
        o[:, 0] = 1.0 / torch.sqrt(K)
        return o

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map:  exp_x(v)  :  T_x M → M

        exp_x(v) = cosh(‖v‖_L · √K) · x  +  sinh(‖v‖_L · √K) · v / ‖v‖_L

        Computed in float64; proj applied in float64 before dtype cast
        to prevent Precision Boundary Rupture on downcast.
        """
        orig_dtype = x.dtype
        K64 = self.K.double().to(x.device)
        x64, v64 = x.double(), v.double()

        v_norm_sq = torch.abs(self.minkowski_inner(v64, v64)) + self.eps
        v_norm = torch.sqrt(v_norm_sq)
        scale = (v_norm * torch.sqrt(K64)).clamp(max=15.0)

        cosh_s = torch.cosh(scale)
        sinh_s = torch.sinh(scale) / (v_norm + self.eps)

        result64 = cosh_s * x64 + sinh_s * v64
        result64 = self.proj(result64)          # constraint enforced at f64
        return result64.to(orig_dtype)

    def exp_map_zero(self, v: torch.Tensor, max_tangent_norm: float = 1.5) -> torch.Tensor:
        """
        Exponential map from the origin (fast path for embedding lookup).

        v must satisfy v[..., 0] = 0  (valid tangent at origin).

        FIX: clamp the spatial tangent L2 norm to max_tangent_norm before
        computing cosh/sinh. This prevents radius explosion across transformer
        layers without breaking the exp/log roundtrip (log_map never calls
        exp_map_zero, so clamping here does not affect roundtrip correctness).

        Root cause: HyperbolicResidualV2 accumulates spatial norms across layers.
        Each layer maps to tangent (log_map_zero), modifies, then lifts back
        (exp_map_zero). Without a norm ceiling, ||v_spatial|| grows from ~2 (emb)
        to ~13.5 after 4 layers — radius = ||v_spatial|| for K=1.

        max_tangent_norm=4.0 → geodesic radius ≤ 4.0 per exp_map_zero call.
        After residual accumulation across 4 layers the effective radius ≈ 2-3,
        which is a healthy operating region for 128-dim hyperbolic space.
        """
        K = self.K.to(v.device, v.dtype)
        sqrt_K = torch.sqrt(K)

        v_spatial = v[..., 1:]
        v_spatial = torch.nan_to_num(v_spatial, nan=0.0, posinf=0.0, neginf=0.0)

        # FIX: clamp L2 norm (not per-element) before cosh/sinh
        # Per-element clamp (old: v_norm.clamp(max=15)) allows huge norms on
        # high-dim vectors: sqrt(128) * 15 = 170 per-element → sinh(170) = inf.
        # L2 norm clamp preserves direction, kills explosion at the source.
        raw_norm = torch.norm(v_spatial, dim=-1, keepdim=True).clamp(min=self.eps)
        v_spatial = torch.where(
            raw_norm > max_tangent_norm,
            v_spatial * (max_tangent_norm / raw_norm),
            v_spatial,
        )

        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True).clamp(min=self.eps)
        scale = (sqrt_K * v_norm).clamp(max=15.0)

        cosh_s = torch.cosh(scale)
        sinh_s = torch.sinh(scale) / (v_norm + self.eps)

        x_time = cosh_s / sqrt_K
        x_spatial = sinh_s * v_spatial

        return self.proj(torch.cat([x_time, x_spatial], dim=-1))

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map:  log_x(y)  :  M → T_x M

        Returns float64 — callers must cast explicitly if they need float32.

        Two corrections applied (TB-PAG):
        1. K_eff = -1/<x64,x64>_L  so that <x64, u>_L = 0 exactly in f64.
        2. No downcast to orig_dtype — f32 rounding yields tangency error
           |<x_f32, δ>_L| ~ 2e-5 which exceeds tolerance.
        """
        orig_dtype = x.dtype
        K64 = self.K.double().to(x.device)

        x64 = x.double()
        y64 = self.proj(y.double())   # reproject y to restore exact y₀

        # K_eff: curvature implicit in received x (avoids float32→float64 error)
        inn_xx = self.minkowski_inner(x64, x64)      # [..., 1] float64
        K_eff = -1.0 / inn_xx.clamp(max=-1e-8)

        # Directional vector in T_x M  (⟨x64, u64⟩_L = 0 exactly)
        inn_xy = self.minkowski_inner(x64, y64)
        u64 = y64 + K_eff * inn_xy * x64

        # Geodesic distance
        d64 = self.dist(x64.to(orig_dtype), y64.to(orig_dtype)).double().unsqueeze(-1)

        # Lorentzian norm of u (spacelike ≥ 0)
        u_norm = torch.sqrt(
            self.minkowski_inner(u64, u64).clamp(min=0.0) + 1e-30
        )

        # Stable scaling: v = (d / ‖u‖_L) · u
        scale = torch.where(d64 > 1e-7, d64 / u_norm, torch.ones_like(d64))
        return scale * u64   # float64 — do NOT cast here

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin — primary entry for LM head and layer norm.

        Returns float64.  Caller casts to working dtype after inspection.
        """
        o = self.origin(
            batch_size=y.shape[0] if y.dim() >= 2 else 1,
            device=y.device,
            dtype=y.dtype,
        )
        # Broadcast origin to match y's batch shape
        if y.dim() == 3:
            # [B, L, n+1]  →  origin needs to match first dim = B*L
            BL = y.shape[0] * y.shape[1]
            y_flat = y.reshape(BL, y.shape[-1])
            o_flat = self.origin(BL, device=y.device, dtype=y.dtype)
            result = self.log_map(o_flat, y_flat)
            return result.reshape(y.shape[0], y.shape[1], -1)
        return self.log_map(o, y)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance:  d(x,y) = (1/√K) · arccosh(-K · <x,y>_L)

        Computed in float64; returns in caller's original dtype.
        """
        orig_dtype = x.dtype
        x64 = self.proj(x.double())
        y64 = self.proj(y.double())
        K64 = self.K.to(x.device, torch.float64)

        inner = self.minkowski_inner(x64, y64).squeeze(-1)
        arg = (-K64 * inner).clamp(min=1.0 + self.eps, max=1e5)
        d = safe_acosh_v2(arg) / torch.sqrt(K64)
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return d.to(orig_dtype)

    # ── Diagnostic ────────────────────────────────────────────────────────

    def manifold_violation(self, x: torch.Tensor) -> torch.Tensor:
        """Mean absolute Minkowski constraint error: mean |<x,x>_L + 1/K|."""
        if not torch.isfinite(x).all():
            return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        K = self.K.to(x.device, x.dtype)
        inner = self.minkowski_inner(x, x).squeeze(-1)
        target = -1.0 / K
        err = (inner - target).abs()
        return torch.nan_to_num(err, nan=0.0).mean()

    def check_upper_sheet(self, x: torch.Tensor) -> bool:
        """Return True iff all points have x₀ > 0 (upper sheet constraint)."""
        return bool((x[..., 0] > 0).all().item())

    # ── Riemannian residual (replaces Euclidean `x + delta`) ─────────────

    def riemannian_add(self, x: torch.Tensor, v_euclidean: torch.Tensor) -> torch.Tensor:
        """
        Geometrically correct 'addition' of a Euclidean displacement v_euclidean
        to a manifold point x.

        Algorithm:
            1. log_map_zero(x) → tangent vector u at origin
            2. u_spatial += step * v_euclidean_spatial  (tangent addition OK)
            3. exp_map_zero(u) → new manifold point
            4. proj() for numerical safety

        This is the correct replacement for the illegal:  x = x + v
        """
        # Map x to tangent at origin
        u = self.log_map_zero(x).to(x.dtype)          # [..., n+1]
        u_spatial = u[..., 1:]                         # [..., n]

        # v_euclidean must match spatial dimension
        v_sp = v_euclidean
        if v_sp.shape[-1] == self.n + 1:
            v_sp = v_sp[..., 1:]                       # strip time if ambient

        # Addition in tangent space (valid — tangent is a linear space)
        u_spatial_new = u_spatial + v_sp

        # Reconstruct full tangent vector with zero time component
        v_time = torch.zeros_like(u_spatial_new[..., :1])
        u_new = torch.cat([v_time, u_spatial_new], dim=-1)

        # Map back to manifold and project
        x_new = self.exp_map_zero(u_new)
        return self.proj(x_new)

    # ── Parallel transport ────────────────────────────────────────────────

    def parallel_transport_zero(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport tangent vector v ∈ T_o M  to  T_y M.
        """
        K = self.K.to(v.device, v.dtype)
        o = self.origin(1, device=v.device, dtype=v.dtype).squeeze(0)

        inner_v_y = self.minkowski_inner(v, y)
        inner_o_y = self.minkowski_inner(o.unsqueeze(0), y)
        denom = (1.0 - K * inner_o_y).clamp(min=self.eps)
        coeff = K * inner_v_y / denom

        return v - coeff * (o + y)
