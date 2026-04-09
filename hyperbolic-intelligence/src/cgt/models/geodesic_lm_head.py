# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
geodesic_lm_head.py
===================
GeodesicLMHeadV2 — LM head using negative squared geodesic distance scoring.

Why this exists (vs HyperbolicLMHeadV2 in lm_head_v2.py):
    HyperbolicLMHeadV2 uses Minkowski inner product scoring:
        logit_k = τ · ⟨h, w_k⟩_L
    This is fast but blind to radius at large r — ∂L/∂r grows with r,
    incentivizing the DegEq Geometric Route.

    GeodesicLMHeadV2 uses negative squared geodesic distance:
        logit_k = -τ · d_H(h, w_k)²
    Gradient w.r.t. radius decays as r → ∞ (verified empirically:
    ∂L/∂r = 0.145 @ r=1.5, 0.002 @ r=10.0 vs Minkowski -2.341, -15.820).
    This provides implicit radius regularization through the LM objective.

Known tradeoff (from benchmark):
    Forward:  18.2ms vs 12.4ms Minkowski  (+47% overhead, CPU)
    Backward: 39.5ms vs 25.1ms Minkowski  (+57% overhead, CPU)
    The overhead comes from safe_acosh_v2 over V=50257 tokens.
    On GPU with fused kernels the gap narrows — re-benchmark before
    deciding to use in production.

Validation:
    torch.autograd.gradcheck passes with atol=1e-4, eps=1e-6.
    Test in: src/cgt/tests/validation_tests.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cgt.geometry.lorentz_v2 import safe_acosh_v2


def _sanitize(x: torch.Tensor, name: str = "") -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class GeodesicLMHeadV2(nn.Module):
    """
    LM head scoring via negative squared geodesic distance on H^n.

    logit_k = -clamp(τ, 0.01, 2.5) · d_H(h, w_k)²

    where d_H uses the safe_acosh_v2 Taylor expansion near x=1 to
    avoid gradient explosion for nearby points.

    Vocab weights are stored as tangent vectors [V, n] (zero-padded to
    [V, n+1]) and lifted to H^n via exp_map_zero at forward time —
    same symmetric parameterization as HyperbolicLMHeadV2 (Variant E).
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        substrate,
        tie_weights: bool = False,
        input_embeddings: nn.Embedding | None = None,
        init_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.substrate = substrate
        self.ambient_dim = substrate.n + 1

        init_temp = init_logit_scale / math.sqrt(max(n_embd, 1))
        self.logit_scale = nn.Parameter(torch.tensor(init_temp))

        if tie_weights and input_embeddings is not None:
            self._tied_embeddings = input_embeddings
            self.weight = None
        else:
            # Store as tangent vectors [V, n] — lifted at forward time
            self.weight = nn.Parameter(
                torch.randn(vocab_size, n_embd) * 0.01
            )
            self._tied_embeddings = None

    def _get_tangent_weight(self) -> torch.Tensor:
        """Return vocab tangent vectors [V, n]."""
        if self._tied_embeddings is not None:
            return self._tied_embeddings.weight  # [V, n_embd]
        return self.weight  # [V, n]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, n+1]  Lorentz points on H^n
        Returns:
            logits: [B, L, V]
        """
        hidden_states = _sanitize(hidden_states, "geo_lm_input")

        # Lift vocab from tangent to manifold
        v = self._get_tangent_weight()                          # [V, n]
        v0 = torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)
        v_full = torch.cat([v0, v], dim=-1)                    # [V, n+1]
        w_vocab = self.substrate.exp_map_zero(v_full)          # [V, n+1]

        # Compute geodesic distance: [B, L, V]
        # Broadcasting: hidden [B, L, 1, n+1] vs vocab [1, 1, V, n+1]
        h_exp = hidden_states.unsqueeze(-2)                    # [B, L, 1, n+1]
        w_exp = w_vocab.unsqueeze(0).unsqueeze(0)              # [1, 1, V, n+1]

        # d_H in float64 via substrate.dist
        d = self.substrate.dist(h_exp, w_exp)                  # [B, L, V]
        d_sq = d.pow(2)

        # Negative squared distance scoring
        tau = self.logit_scale.clamp(0.01, 2.5)
        logits = -tau * d_sq

        return _sanitize(logits.to(hidden_states.dtype), "geo_lm_logits")


class AngularLMHead(nn.Module):
    """
    AngularLMHead — purely angular scoring for hyperbolic LM head.

    Motivation (V5 / HyDRA v3 Channel 2 fix):
        GeodesicLMHeadV2 computes scores via geodesic distances or Minkowski
        inner products, both of which have ∂score/∂r ≠ 0. This creates a
        persistent radial gradient path from the CE/KL loss that bypasses
        any loss-side intervention (D1, D3, OTED, etc.) and sustains the
        DegEq attractor as a second independent channel.

        AngularLMHead severs this channel by:
          1. Projecting h and W to T_o H^n via log_map_zero
          2. Normalising to unit vectors on S^(n-1)
          3. Scoring via cosine similarity

        Result: ∂score/∂r = 0 exactly for all r, all k.
        The only radial signal comes from the explicit anchor in the loss
        (OTED or D3), which is by design.

    Trade-off:
        Loses the distance-based metric structure of the hyperbolic scoring.
        All tokens at the same angle score equally regardless of radius.
        This is the desired property when debugging DegEq; it may degrade
        language modeling quality if angular resolution is insufficient.

    Usage:
        head = AngularLMHead(n_embd=128, vocab_size=50257, substrate=substrate)
        logits = head(hidden_states)  # [B, L, V]

    Activation in DistillationConfigV2:
        use_angular_head = True
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        substrate,
        init_logit_scale: float = 1.0,
        tie_weights: bool = False,
        input_embeddings: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.n_embd      = n_embd
        self.vocab_size  = vocab_size
        self.substrate   = substrate
        self.ambient_dim = substrate.n + 1

        init_temp = init_logit_scale / math.sqrt(max(n_embd, 1))
        self.logit_scale = nn.Parameter(torch.tensor(init_temp))

        if tie_weights and input_embeddings is not None:
            self._tied = input_embeddings
            self.weight = None
        else:
            # Store as tangent vectors [V, n] — same convention as GeodesicLMHeadV2
            self.weight = nn.Parameter(
                torch.randn(vocab_size, n_embd) * 0.01
            )
            self._tied = None

    def _get_tangent_weight(self) -> torch.Tensor:
        if self._tied is not None:
            return self._tied.weight
        return self.weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_step: int = 10000,
        warmup_steps: int = 500,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, n+1]  points on H^n
            global_step:   current training step (for tau warmup)
            warmup_steps:  ramp tau from 0.1 to logit_scale over this many steps
        Returns:
            logits: [B, L, V]  cosine similarities in T_o, ∂logit/∂r = 0

        Warmup rationale:
            Without warmup, cosine similarity produces very sharp gradients
            early in training (all tokens on unit sphere equidistant),
            causing DegEq onset acceleration (observed in V5-C run, step 300).
            Linear ramp from tau=0.1 to tau=logit_scale over warmup_steps
            prevents early gradient explosion.
        """
        B, L, _ = hidden_states.shape

        # ── Project h to T_o via log_map_zero, take spatial slice ──────────
        # log_map_zero(h) = (0, r_h * û_h) in ambient coords
        # spatial slice [1:] gives v_h = r_h * û_h ∈ R^n
        h_log = self.substrate.log_map_zero(
            hidden_states.reshape(-1, hidden_states.shape[-1])
        )[:, 1:]                                        # [BL, n]

        # ── Normalise to unit sphere S^{n-1} ── ∂û/∂r = 0 exactly ─────────
        u_h = torch.nn.functional.normalize(h_log, dim=-1, eps=1e-7)  # [BL, n]

        # ── Vocab: lift tangent [V,n] to T_o, normalise ────────────────────
        v_w = self._get_tangent_weight()                # [V, n]
        u_w = torch.nn.functional.normalize(v_w, dim=-1, eps=1e-7)  # [V, n]

        # ── Cosine similarity ───────────────────────────────────────────────
        # Warmup: ramp tau from 0.1 → logit_scale over warmup_steps
        warmup_frac = min(1.0, global_step / max(warmup_steps, 1))
        tau_target  = self.logit_scale.clamp(0.01, 2.5)
        tau         = 0.1 + (tau_target - 0.1) * warmup_frac
        u_w    = u_w.to(u_h.dtype)                    # match float64/float32
        logits = (u_h @ u_w.T) / tau                   # [BL, V]

        return torch.nan_to_num(
            logits.reshape(B, L, -1).to(hidden_states.dtype),
            nan=0.0, posinf=1e4, neginf=-1e4
        )
