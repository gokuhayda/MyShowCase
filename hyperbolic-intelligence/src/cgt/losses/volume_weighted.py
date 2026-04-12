# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Hyperbolic Volume-Weighted Loss
================================

In H^n with curvature K, the Riemannian volume element at geodesic
radius r is:

    dV_H(r) = (sinh(r√K) / √K)^{n-1} dr dΩ

This is the Jacobian determinant of the exponential map exp_o.
It grows exponentially with r, meaning:

    - Near the origin (r ≈ 0): volume is small → tokens are DENSE
    - Near the boundary (r → ∞): volume is exponential → tokens are SPARSE

Standard losses (CE, KL, MSE) treat all tokens equally regardless of
their manifold position. This ignores the geometry: a token at r = 5
occupies ~sinh(5)^127 ≈ 10^270 times more volume than a token at r = 0.1.

DegEq equalizes all radii to r ≈ some fixed value, which WASTES the
exponential volume available at the boundary. A volume-weighted loss
assigns importance inversely proportional to available volume:

    w(r) = 1 / dV_H(r)  (normalized)

This gives HIGH weight to tokens near the origin (dense, important,
frequent) and LOW weight to tokens at the boundary (sparse, rare).
Crucially: if DegEq tries to push all tokens to the same radius,
the volume weights become uniform → the loss no longer benefits from
the geometric prior → gradient pressure pushes tokens BACK to their
natural radial positions.

The volume weight is the INVERSE of the Jacobian determinant of the
hyperbolic exponential map — the same mathematical object that the
P4 network document uses as a proxy for informational density.

Usage:
    from cgt.losses.volume_weighted import VolumeWeightedCE, VolumeWeightedKL

    # Drop-in replacement for nn.CrossEntropyLoss
    loss_fn = VolumeWeightedCE(substrate=substrate, strength=0.5)
    loss = loss_fn(logits, targets, hidden_states)

    # Volume-weighted KL for distillation
    kl_fn = VolumeWeightedKL(substrate=substrate, temperature=3.0)
    loss = kl_fn(student_logits, teacher_logits, hidden_states)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def hyperbolic_volume_weight(
    hidden_states: torch.Tensor,
    K: float = 1.0,
    n_dim: Optional[int] = None,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute per-token weight inversely proportional to the local
    hyperbolic volume element.

    The volume element at radius r on H^n_K is:
        dV(r) ∝ sinh(r√K)^{n-1}

    The weight is the INVERSE (high weight near origin, low at boundary):
        w(r) = 1 / (sinh(r√K)^{n-1} + ε)

    For numerical stability with large n, we work in log-space:
        log w(r) = -(n-1) * log(sinh(r√K) + ε)

    Args:
        hidden_states: [..., D+1] Lorentz ambient coordinates.
            x[..., 0] = time coordinate, x[..., 1:] = spatial.
        K: Curvature parameter (default 1.0).
        n_dim: Intrinsic dimension n. If None, inferred as D+1-1 = D.
        eps: Numerical floor.
        normalize: If True, normalize weights to sum to batch size
            (preserves loss magnitude scale).

    Returns:
        weights: [...] per-token volume weights.
    """
    # Geodesic radius: r = ||x_spatial||
    spatial = hidden_states[..., 1:]
    r = spatial.norm(dim=-1).clamp(min=eps)

    if n_dim is None:
        n_dim = spatial.shape[-1]  # intrinsic dim = spatial dim

    sqK = math.sqrt(K)

    # Log-space for stability: log w = -(n-1) * log(sinh(r*sqK))
    sinh_r = torch.sinh(r * sqK).clamp(min=eps)
    log_w = -(n_dim - 1) * torch.log(sinh_r)

    # Shift for numerical stability (subtract max before exp)
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)

    if normalize:
        # Normalize so that mean weight = 1 (preserves loss scale)
        w = w * (w.numel() / w.sum().clamp(min=eps))

    return w.detach()  # No gradient through weights


class VolumeWeightedCE(nn.Module):
    """
    Cross-entropy loss weighted by inverse hyperbolic volume.

    L = (1/N) Σ_i w(r_i) * CE(logits_i, target_i)

    where w(r) ∝ 1/sinh(r√K)^{n-1}.

    Tokens near the origin (frequent, high-density) get higher weight.
    Tokens at the boundary (rare, low-density) get lower weight.

    DegEq resistance: if all tokens collapse to the same radius,
    all weights become equal → loss degrades to unweighted CE →
    the model loses the geometric advantage and gradient pressure
    pushes tokens back toward differentiated radii.

    Args:
        substrate: LorentzSubstrate (optional, for K extraction).
        K: Curvature (default 1.0, overridden by substrate).
        strength: Interpolation with uniform weights.
            0.0 = standard CE (no volume weighting).
            1.0 = full volume weighting.
            0.5 = blend (recommended starting point).
        ignore_index: Token index to ignore (e.g. padding).
    """

    def __init__(
        self,
        substrate=None,
        K: float = 1.0,
        strength: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        if substrate is not None and hasattr(substrate, 'K'):
            K_val = substrate.K
            self.K = K_val.item() if hasattr(K_val, 'item') else float(K_val)
        else:
            self.K = K
        self.strength = strength
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L, V] vocabulary logits.
            targets: [B, L] target token indices.
            hidden_states: [B, L, D+1] Lorentz-ambient hidden states.

        Returns:
            Scalar weighted cross-entropy loss.
        """
        B, L, V = logits.shape

        # Per-token CE (unreduced)
        ce = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction='none',
            ignore_index=self.ignore_index,
        )  # [B*L]

        # Volume weights
        vol_w = hyperbolic_volume_weight(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            K=self.K,
        )  # [B*L]

        # Blend with uniform
        uniform = torch.ones_like(vol_w)
        w = self.strength * vol_w + (1.0 - self.strength) * uniform

        # Mask ignored tokens
        mask = (targets.reshape(-1) != self.ignore_index).float()
        w = w * mask

        # Weighted mean
        return (w * ce).sum() / w.sum().clamp(min=1e-8)


class VolumeWeightedKL(nn.Module):
    """
    KL divergence distillation loss weighted by inverse hyperbolic volume.

    L = (1/N) Σ_i w(r_i) * KL(student_i || teacher_i)

    For distillation with temperature T:
        KL = T² * Σ_k p_teacher(k) * log(p_teacher(k) / p_student(k))

    Volume weighting ensures that tokens in the dense manifold region
    (near origin, high-frequency) contribute more to the distillation
    signal than tokens in the sparse boundary region.

    Args:
        substrate: LorentzSubstrate (optional).
        K: Curvature (default 1.0).
        temperature: Distillation temperature (default 3.0).
        strength: Volume weighting strength (0=uniform, 1=full).
    """

    def __init__(
        self,
        substrate=None,
        K: float = 1.0,
        temperature: float = 3.0,
        strength: float = 0.5,
    ):
        super().__init__()
        if substrate is not None and hasattr(substrate, 'K'):
            K_val = substrate.K
            self.K = K_val.item() if hasattr(K_val, 'item') else float(K_val)
        else:
            self.K = K
        self.temperature = temperature
        self.strength = strength

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, L, V] student vocabulary logits.
            teacher_logits: [B, L, V] teacher vocabulary logits.
            hidden_states: [B, L, D+1] student Lorentz-ambient states.

        Returns:
            Scalar weighted KL divergence loss.
        """
        T = self.temperature
        B, L, V = student_logits.shape

        # Soft distributions
        p_student = F.log_softmax(student_logits / T, dim=-1)
        p_teacher = F.softmax(teacher_logits / T, dim=-1)

        # Per-token KL (sum over vocab, keep token dimension)
        kl = F.kl_div(
            p_student.reshape(-1, V),
            p_teacher.reshape(-1, V),
            reduction='none',
        ).sum(dim=-1)  # [B*L]

        kl = kl * (T * T)  # Standard distillation scaling

        # Volume weights
        vol_w = hyperbolic_volume_weight(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            K=self.K,
        )

        # Blend
        uniform = torch.ones_like(vol_w)
        w = self.strength * vol_w + (1.0 - self.strength) * uniform

        return (w * kl).sum() / w.sum().clamp(min=1e-8)


class VolumeWeightedMSE(nn.Module):
    """
    MSE hidden-state alignment loss weighted by inverse hyperbolic volume.

    L = (1/N) Σ_i w(r_i) * ||student_i - target_i||²

    Useful for the λ_hidden term in the HyDRA distillation objective,
    where hidden-state alignment in the dense manifold region matters
    more than alignment at the sparse boundary.

    Args:
        substrate: LorentzSubstrate (optional).
        K: Curvature (default 1.0).
        strength: Volume weighting strength.
    """

    def __init__(
        self,
        substrate=None,
        K: float = 1.0,
        strength: float = 0.5,
    ):
        super().__init__()
        if substrate is not None and hasattr(substrate, 'K'):
            K_val = substrate.K
            self.K = K_val.item() if hasattr(K_val, 'item') else float(K_val)
        else:
            self.K = K
        self.strength = strength

    def forward(
        self,
        student_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_hidden: [B, L, D+1] student hidden states (Lorentz).
            target_hidden: [B, L, D'] target hidden states (projected).

        Returns:
            Scalar weighted MSE loss.
        """
        # MSE per token
        # Align dims if needed (target may be different D)
        min_d = min(student_hidden.shape[-1], target_hidden.shape[-1])
        diff = student_hidden[..., :min_d] - target_hidden[..., :min_d]
        mse = (diff * diff).sum(dim=-1)  # [B, L]

        # Volume weights from student positions
        vol_w = hyperbolic_volume_weight(
            student_hidden.reshape(-1, student_hidden.shape[-1]),
            K=self.K,
        ).reshape(mse.shape)

        # Blend
        uniform = torch.ones_like(vol_w)
        w = self.strength * vol_w + (1.0 - self.strength) * uniform

        return (w * mse).sum() / w.sum().clamp(min=1e-8)
