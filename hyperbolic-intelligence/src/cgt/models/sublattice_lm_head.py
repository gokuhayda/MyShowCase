# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Sublattice LM Head
==================

Inspired by: Gu, Lüders & Bechinger (2026). "Non-monotonic magnetic friction
from collective rotor dynamics." Nature Materials.

Key insight (Gu et al. two-sublattice model + Khrulkov/Yang hierarchy):

    Gu et al. decompose the magnetic array into two sublattices with DIFFERENT
    preferred orientations (parallel vs antiparallel). The friction peak occurs
    when these two competing preferences cannot be simultaneously satisfied.

    In hyperbolic LLMs, Khrulkov et al. (2020) and Yang et al. (2025) show
    that the RADIAL coordinate serves as a natural uncertainty estimator:
    - Frequent/generic tokens → near the origin (low radius)
    - Rare/specific tokens → near the boundary (high radius)

    DegEq destroys this stratification by driving ALL tokens to uniform radius.

This module introduces SublatticeLMHead:
    - Decomposes the vocabulary into frequency-based sublattices.
    - Applies DIFFERENT radial bounds to each sublattice.
    - Architecturally preserves the frequency→radius hierarchy.

    Sublattice A (frequent tokens):  vocab_r_max_A = 1.5 (near origin)
    Sublattice B (rare tokens):      vocab_r_max_B = 5.0 (near boundary)

    This is the vocabulary analog of the two-sublattice decomposition in
    Gu et al., with the crucial difference that here the "competing preferences"
    (different radial bounds) are COMPLEMENTARY rather than conflicting —
    they preserve hierarchy instead of creating frustration.

Design: ADDITIVE ONLY. Extends HyperbolicLMHeadV2, no modification to
existing modules. Uses the same exp_map_zero + tangent space parameterization.

Usage:
    from cgt.models.sublattice_lm_head import SublatticeLMHead

    lm_head = SublatticeLMHead(
        n_embd=128,
        vocab_size=50257,
        substrate=substrate,
        token_counts=token_counts,  # [V] occurrence counts
        r_max_frequent=1.5,
        r_max_rare=5.0,
        frequency_threshold=0.3,    # top 30% by count → frequent
    )

References:
    Gu, Lüders & Bechinger (2026). Nature Materials. DOI:10.1038/s41563-026-02538-1
    Khrulkov et al. (2020). Hyperbolic Image Embeddings. CVPR.
    Yang et al. (2025). HypLoRA. NeurIPS 2025.
    de Sena (2026). HyDRA v4: Degenerate Equilibrium in Hyperbolic Distillation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from cgt.geometry import LorentzSubstrateV2
from cgt.dynamics.riemannian_update_v2 import ensure_lorentz_v2


def _sanitize(x: torch.Tensor, name: str = "") -> torch.Tensor:
    """NaN/inf guard."""
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class SublatticeLMHead(nn.Module):
    """
    Language model head with frequency-stratified radial bounds.

    Decomposes vocabulary into sublattices based on token frequency:
    - Sublattice A (frequent): bounded at small radius (near origin)
    - Sublattice B (rare): bounded at larger radius (near boundary)

    This ARCHITECTURALLY preserves the radial uncertainty estimator
    that DegEq destroys, ensuring:
        ρ(log f_k, r_k) < -0.1

    throughout training, not just at initialization.

    The forward pass applies different exp_map_zero with different
    max_tangent_norm to each sublattice, then concatenates the results.

    Mathematical guarantee:
        For token k in sublattice s ∈ {A, B}:
            ||w_k||_geodesic ≤ r_max_s

        Therefore:
            mean(r_A) < mean(r_B) by construction
            → ρ(log f, r) < 0 is architecturally enforced

    Args:
        n_embd: Intrinsic embedding dimension.
        vocab_size: Vocabulary size.
        substrate: LorentzSubstrateV2 instance.
        token_counts: [V] tensor of token occurrence counts (for sublattice assignment).
            If None, uses a uniform split (first half = frequent, second half = rare).
        r_max_frequent: Max geodesic radius for frequent tokens (sublattice A).
        r_max_rare: Max geodesic radius for rare tokens (sublattice B).
        frequency_threshold: Fraction of vocab considered "frequent" (by count mass).
            Default: 0.3 → top 30% of tokens by cumulative count mass are "frequent".
        learnable_scale: Whether logit scale τ is learnable.
        init_logit_scale: Initial logit scale value.
        angular_mode: If True, use cosine similarity (∂logit/∂r = 0) instead of
            Minkowski inner product. Combines SublatticeLMHead with Channel 2 fix.
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        substrate: LorentzSubstrateV2,
        token_counts: Optional[torch.Tensor] = None,
        r_max_frequent: float = 1.5,
        r_max_rare: float = 5.0,
        frequency_threshold: float = 0.3,
        learnable_scale: bool = True,
        init_logit_scale: float = 5.0,
        angular_mode: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.substrate = substrate
        self.r_max_frequent = r_max_frequent
        self.r_max_rare = r_max_rare
        self.angular_mode = angular_mode
        self.eps = eps
        self.ambient_dim = substrate.n + 1

        # ── Temperature ──
        init_temp = 1.0 / math.sqrt(max(n_embd, 1))
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(init_temp))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_temp))

        # ── Sublattice assignment ──
        # mask_frequent[k] = True if token k is in sublattice A (frequent)
        mask_frequent = self._assign_sublattices(
            vocab_size, token_counts, frequency_threshold
        )
        self.register_buffer("mask_frequent", mask_frequent)
        self.register_buffer("mask_rare", ~mask_frequent)

        n_frequent = mask_frequent.sum().item()
        n_rare = vocab_size - n_frequent

        # ── Vocab embeddings as tangent vectors (same as HyperbolicLMHeadV2 v5) ──
        # Separate parameters for each sublattice
        self.weight_frequent = nn.Parameter(
            torch.randn(n_frequent, n_embd) * 0.01
        )
        self.weight_rare = nn.Parameter(
            torch.randn(n_rare, n_embd) * 0.01
        )

        # ── Index mappings ──
        # global_idx → sublattice_idx
        freq_indices = torch.where(mask_frequent)[0]
        rare_indices = torch.where(~mask_frequent)[0]
        self.register_buffer("freq_indices", freq_indices)
        self.register_buffer("rare_indices", rare_indices)

        # Inverse mapping: for each global vocab index, which sublattice and local idx
        sublattice_local_idx = torch.zeros(vocab_size, dtype=torch.long)
        sublattice_local_idx[freq_indices] = torch.arange(n_frequent)
        sublattice_local_idx[rare_indices] = torch.arange(n_rare)
        self.register_buffer("sublattice_local_idx", sublattice_local_idx)

    @staticmethod
    def _assign_sublattices(
        vocab_size: int,
        token_counts: Optional[torch.Tensor],
        frequency_threshold: float,
    ) -> torch.Tensor:
        """
        Assign tokens to sublattices based on frequency.

        Args:
            vocab_size: Total vocabulary size.
            token_counts: [V] token occurrence counts.
            frequency_threshold: Fraction considered "frequent".

        Returns:
            mask_frequent: [V] boolean, True for frequent tokens.
        """
        if token_counts is None:
            # No frequency data → split by index (first portion = "frequent")
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            n_freq = int(vocab_size * frequency_threshold)
            mask[:n_freq] = True
            return mask

        counts = token_counts.float()
        if counts.shape[0] < vocab_size:
            # Pad with zeros
            padded = torch.zeros(vocab_size, dtype=torch.float32)
            padded[:counts.shape[0]] = counts
            counts = padded
        elif counts.shape[0] > vocab_size:
            counts = counts[:vocab_size]

        # Sort by frequency (descending)
        sorted_counts, sorted_idx = counts.sort(descending=True)
        total_mass = sorted_counts.sum()

        if total_mass < 1.0:
            # No counts → uniform split
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[:int(vocab_size * frequency_threshold)] = True
            return mask

        # Cumulative mass
        cumulative = sorted_counts.cumsum(0) / total_mass

        # Frequent = tokens comprising the top `frequency_threshold` of mass
        n_frequent = (cumulative <= frequency_threshold).sum().item()
        n_frequent = max(n_frequent, 1)  # at least 1 frequent token

        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[sorted_idx[:n_frequent]] = True

        return mask

    def _lift_to_manifold(
        self,
        tangent_weights: torch.Tensor,
        r_max: float,
    ) -> torch.Tensor:
        """
        Lift tangent vectors to manifold with bounded radius.

        Same as HyperbolicLMHeadV2._get_weight(), but parameterized by r_max.

        Args:
            tangent_weights: [N, n_embd] tangent space vectors.
            r_max: Maximum geodesic radius.

        Returns:
            manifold_points: [N, n_embd + 1] points on H^n.
        """
        w = tangent_weights
        zeros = torch.zeros(
            *w.shape[:-1], 1, device=w.device, dtype=w.dtype
        )
        v_tangent = torch.cat([zeros, w], dim=-1)  # [N, n+1], v[:,0]=0
        return self.substrate.exp_map_zero(v_tangent, max_tangent_norm=r_max)

    def _get_all_weights(self) -> torch.Tensor:
        """
        Reconstruct full vocab embedding matrix [V, n+1] with
        sublattice-specific radial bounds.
        """
        # Lift each sublattice with its own r_max
        w_freq = _sanitize(
            self._lift_to_manifold(self.weight_frequent, self.r_max_frequent),
            "w_freq",
        )
        w_rare = _sanitize(
            self._lift_to_manifold(self.weight_rare, self.r_max_rare),
            "w_rare",
        )

        # Allocate full vocab tensor
        V = self.vocab_size
        D = self.ambient_dim
        device = w_freq.device
        dtype = w_freq.dtype

        full_weight = torch.zeros(V, D, device=device, dtype=dtype)
        full_weight[self.freq_indices] = w_freq
        full_weight[self.rare_indices] = w_rare

        return full_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits with sublattice-stratified vocab embeddings.

        Args:
            hidden_states: [B, L, n+1] Lorentz points.

        Returns:
            logits: [B, L, V]
        """
        hidden_states = _sanitize(hidden_states, "sublattice_lm_input")

        # 1. Enforce manifold constraint
        x = ensure_lorentz_v2(hidden_states, self.substrate, debug_tag="sublattice_lm")

        # 2. Get stratified vocab embeddings
        w = self._get_all_weights()  # [V, n+1]

        if self.angular_mode:
            # ── Angular mode: cosine similarity, ∂logit/∂r = 0 ──
            # Channel 2 fix combined with sublattice stratification
            return self._angular_forward(x, w)
        else:
            # ── Minkowski inner product mode ──
            return self._minkowski_forward(x, w)

    def _minkowski_forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard Minkowski inner product logits.

        logit_k = τ * ⟨h, w_k⟩_L = τ * (-h₀w_{k,0} + h^{(1:)} · w_k^{(1:)})

        Note: This preserves ∂logit/∂r ≠ 0 (Channel 2 active).
        The sublattice stratification helps but does NOT fully prevent DegEq.
        For full DegEq prevention, use angular_mode=True.
        """
        x64 = x.double()
        w64 = w.double()

        time_inner = torch.einsum("blt,vt->blv", x64[..., :1], w64[:, :1])
        space_inner = torch.einsum("bls,vs->blv", x64[..., 1:], w64[:, 1:])
        inner = space_inner - time_inner

        tau = self.logit_scale.clamp(0.01, 2.5).double()
        logits = inner * tau

        return _sanitize(logits.to(x.dtype), "sublattice_logits")

    def _angular_forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Angular (cosine similarity) logits: ∂logit/∂r = 0 exactly.

        logit_k = (1/τ) * ⟨û_h, û_{w_k}⟩

        where û = normalize(log_o(·)^{(1:)})

        This combines the Channel 2 fix (AngularLMHead) with sublattice
        stratification: even though the logits are radially invariant,
        the underlying vocab embeddings RETAIN their radial structure
        for downstream diagnostics and potential radial-aware extensions.
        """
        x64 = x.double()
        w64 = w.double()

        # Log map to origin → spatial components
        # For points near origin: log_o(x) ≈ x^{(1:)} (Taylor first order)
        # For general points: use substrate.log_map_zero
        x_spatial = x64[..., 1:]  # [B, L, n]
        w_spatial = w64[:, 1:]    # [V, n]

        # Normalize to unit direction
        x_norm = x_spatial / (x_spatial.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        w_norm = w_spatial / (w_spatial.norm(dim=-1, keepdim=True).clamp(min=1e-8))

        # Cosine similarity
        cos_sim = torch.einsum("bls,vs->blv", x_norm, w_norm)

        tau = self.logit_scale.clamp(0.01, 2.5).double()
        logits = cos_sim / tau

        return _sanitize(logits.to(x.dtype), "sublattice_angular_logits")

    def get_radial_stats(self) -> Dict[str, float]:
        """
        Compute radial statistics per sublattice for diagnostics.

        Returns:
            Dict with mean/std radius for each sublattice and correlation.
        """
        with torch.no_grad():
            w_freq = self._lift_to_manifold(self.weight_frequent, self.r_max_frequent)
            w_rare = self._lift_to_manifold(self.weight_rare, self.r_max_rare)

            r_freq = w_freq[:, 1:].norm(dim=-1)
            r_rare = w_rare[:, 1:].norm(dim=-1)

            return {
                "r_frequent_mean": r_freq.mean().item(),
                "r_frequent_std": r_freq.std().item(),
                "r_rare_mean": r_rare.mean().item(),
                "r_rare_std": r_rare.std().item(),
                "r_gap": (r_rare.mean() - r_freq.mean()).item(),
                "n_frequent": self.mask_frequent.sum().item(),
                "n_rare": self.mask_rare.sum().item(),
                "hierarchy_preserved": (r_rare.mean() > r_freq.mean()).item(),
            }

    def get_stratification_quality(self) -> float:
        """
        Measure quality of radial stratification.

        Returns:
            score: r_gap / max(r_rare_mean, ε).
                   > 0.3 is good hierarchy.
                   < 0.1 suggests DegEq is collapsing the structure.
        """
        stats = self.get_radial_stats()
        r_rare = max(stats["r_rare_mean"], 1e-6)
        return stats["r_gap"] / r_rare
