"""
cgt/dynamics/riemannian_update_v2.py
=========================================
Isolated Riemannian phase dynamics for v2.

THE SINGLE CORRECT REPLACEMENT for the Tangent Addition Fallacy:
    ❌  evolved = x + step * phase_signal          # Euclidean addition — ILLEGAL
    ✅  evolved = riemannian_step_v2(x, phase_signal, substrate)

Algorithm (mathematically correct)
-----------------------------------
1.  ensure_lorentz_v2(x)       → x_L   ∈ H^n  shape [..., n+1]
2.  substrate.log_map_zero(x_L)→ v      ∈ T_o   shape [..., n+1]   (float64)
3.  lift_phase_to_tangent(phase_signal, n+1) → ps ∈ T_o, v[0]=0
4.  v' = v + step * ps                 (valid: tangent space is linear)
5.  v'[...,0] = 0                      (enforce tangency at origin)
6.  clamp(v', max_norm)                (numerical stability)
7.  substrate.exp_map_zero(v')         → x'  ∈ H^n
8.  substrate.proj(x')                 (paranoid re-projection)

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry import LorentzSubstrateV2


# ─────────────────────────────────────────────────────────────────────────────
# Shape utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_lorentz_v2(
    x: torch.Tensor,
    substrate: LorentzSubstrateV2,
    debug_tag: str = "",
) -> torch.Tensor:
    """
    Guarantee x has Lorentz-ambient shape [..., n+1] and satisfies the
    Minkowski constraint.

    Case A — x already has shape [..., n+1]:
        proj() recomputes x₀ for numerical exactness.

    Case B — x has Euclidean shape [..., n]:
        Lift by prepending the correct time coordinate:
            x₀ = √(1/K + ‖x‖²)   (satisfies constraint exactly)
        Then proj() for float-safety.

    Raises AssertionError if x.shape[-1] ∉ {n, n+1}.
    """
    n       = substrate.n
    ambient = n + 1
    last    = x.shape[-1]

    if last == ambient:
        return substrate.proj(x)

    if last == n:
        K = substrate.K.to(x.device, x.dtype)
        time = torch.sqrt(
            (1.0 / K + (x ** 2).sum(dim=-1, keepdim=True)).clamp(min=1e-15)
        )
        x_lifted = torch.cat([time, x], dim=-1)
        return substrate.proj(x_lifted)

    raise AssertionError(
        f"ensure_lorentz_v2({debug_tag}): expected last dim {n} (Euclidean) "
        f"or {ambient} (Lorentz-ambient), got {last}. shape={tuple(x.shape)}"
    )


def lift_phase_to_tangent(
    phase_signal: torch.Tensor,
    target_ambient_dim: int,
) -> torch.Tensor:
    """
    Lift a Euclidean phase signal to a valid tangent vector at the origin.

    At o = (1/√K, 0,…,0) the tangent condition is ⟨o, v⟩_L = 0,
    which forces v[...,0] = 0.  So the correct lift is:
        v = cat([0, phase_signal_spatial], dim=-1)

    This is DIFFERENT from ensure_lorentz_v2, which places a point ON the
    manifold.  Here we produce a TANGENT VECTOR — time component must be 0.
    """
    n    = target_ambient_dim - 1
    last = phase_signal.shape[-1]

    if last == target_ambient_dim:
        # Already ambient-shaped → zero out time to enforce tangency
        sig = phase_signal.clone()
        sig[..., 0] = 0.0
        return sig

    if last == n:
        zeros = torch.zeros_like(phase_signal[..., :1])
        return torch.cat([zeros, phase_signal], dim=-1)

    # Mismatch: pad or truncate spatial part, then prepend zero time
    if last < n:
        phase_signal = F.pad(phase_signal, (0, n - last))
    else:
        phase_signal = phase_signal[..., :n]
    zeros = torch.zeros_like(phase_signal[..., :1])
    return torch.cat([zeros, phase_signal], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Core corrected step
# ─────────────────────────────────────────────────────────────────────────────

def riemannian_step_v2(
    x: torch.Tensor,
    phase_signal: torch.Tensor,
    substrate: LorentzSubstrateV2,
    step_size: float = 0.1,
    clamp_tangent: float = 5.0,
) -> torch.Tensor:
    """
    Geometrically correct Riemannian phase update.

    All intermediate tensors are [..., n+1].  No shape crashes.
    Output dtype matches input dtype.

    Steps
    -----
    1. ensure_lorentz_v2(x)   → x_L  ∈ H^n
    2. log_map_zero(x_L)      → v    ∈ T_o  (float64)
    3. lift phase_signal       → ps   ∈ T_o
    4. v' = v + step_size*ps  (tangent addition — valid)
    5. v'[...,0] = 0          (enforce tangency at origin)
    6. clamp v'
    7. exp_map_zero(v')        → x'  ∈ H^n
    8. proj(x')                (paranoid re-projection)
    """
    orig_dtype = x.dtype
    ambient    = substrate.n + 1

    # Step 1 — ensure manifold point
    x_L = ensure_lorentz_v2(x, substrate, debug_tag="riemannian_step_v2.x")

    # Step 2 — map to tangent at origin  (returns float64)
    v = substrate.log_map_zero(x_L)
    v = v.to(dtype=orig_dtype)

    # Step 3 — lift phase signal to tangent format
    ps = lift_phase_to_tangent(phase_signal, ambient).to(dtype=orig_dtype)

    # Step 4 — update in tangent space  (valid linear operation)
    v_updated = v + step_size * ps

    # Step 5 + 6 — enforce tangency & clamp
    v_updated = v_updated.clamp(-clamp_tangent, clamp_tangent)
    v_updated = v_updated.clone()
    v_updated[..., 0] = 0.0

    # Step 7 — map back to manifold
    x_new = substrate.exp_map_zero(v_updated)

    # Step 8 — paranoid re-projection
    x_new = substrate.proj(x_new)

    return x_new   # [..., ambient], dtype=orig_dtype


# ─────────────────────────────────────────────────────────────────────────────
# nn.Module wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RiemannianPhaseDynamicsV2(nn.Module):
    """
    nn.Module wrapping riemannian_step_v2.

    This is the drop-in replacement for the legacy RiemannianPhaseDynamics
    which contained the Tangent Addition Fallacy.

    The `use_riemannian_dynamics=False` path is intentionally REMOVED.
    There is NO Euclidean fallback — if you want no dynamics, skip this module.
    """

    def __init__(
        self,
        substrate: LorentzSubstrateV2,
        step_size: float = 0.1,
        clamp_tangent: float = 5.0,
    ) -> None:
        super().__init__()
        self.substrate     = substrate
        self.step_size     = step_size
        self.clamp_tangent = clamp_tangent

    def forward(
        self,
        embeddings: torch.Tensor,    # [B, L, D] or [B, L, D+1]
        phase_signal: torch.Tensor,  # [B, L, D] or [B, L, D+1]
    ) -> torch.Tensor:               # [B, L, n+1]
        return riemannian_step_v2(
            x            = embeddings,
            phase_signal = phase_signal,
            substrate    = self.substrate,
            step_size    = self.step_size,
            clamp_tangent= self.clamp_tangent,
        )

    def extra_repr(self) -> str:
        return (
            f"step_size={self.step_size}, "
            f"clamp_tangent={self.clamp_tangent}, "
            f"substrate.n={self.substrate.n}"
        )
