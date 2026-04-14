# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Phase-Weighted Attention (GCM paper, eq. 14)
=============================================

Modulates attention weights by Kuramoto phase coherence:

    w_ij = softmax(QKᵀ/√d) · (1 + cos(θᵢ − θⱼ)) / 2

Tokens in phase receive boosted attention; dessynchronised pairs
are down-weighted. This integrates temporal binding directly into
the attention mechanism rather than only as an auxiliary loss.

Reference:
    GCM paper, Section 6 eq. 14:
    "w_ij = 1 + cos(θ_i − θ_j) as perceptual weighting."

Design: drop-in wrapper around existing attention output.
Does NOT change attention computation — multiplies post-softmax weights.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseWeightedAttention(nn.Module):
    """Wraps existing attention output with Kuramoto phase modulation.

    Usage::
        pwa = PhaseWeightedAttention(n_heads=8, strength=0.3)

        # Inside transformer forward:
        attn_out = existing_attention(q, k, v)           # [B, N, D]
        phases   = hakorn_layer.phases                   # [N]
        attn_out = pwa(attn_out, phases)

    Args:
        n_heads:  Number of attention heads (for per-head phase assignment).
        strength: Blend factor α — output = (1-α)·attn + α·phase_weighted.
                  0.0 = disabled, 1.0 = full phase weighting.
                  Default 0.3 (conservative, additive effect).
        learnable_strength: If True, α is a learnable parameter.
    """

    def __init__(
        self,
        n_heads: int = 8,
        strength: float = 0.3,
        learnable_strength: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        if learnable_strength:
            self._alpha = nn.Parameter(torch.tensor(strength))
        else:
            self.register_buffer('_alpha', torch.tensor(strength))

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha.clamp(0.0, 1.0)

    def forward(
        self,
        hidden: torch.Tensor,       # [B, N, D]
        phases: torch.Tensor,       # [N] or [B, N] — Kuramoto phases
        attn_weights: Optional[torch.Tensor] = None,  # [B, H, N, N] optional
    ) -> torch.Tensor:
        """Apply phase modulation to hidden states.

        Args:
            hidden:       Attention output [B, N, D].
            phases:       Kuramoto phase angles [N] or [B, N].
            attn_weights: Raw attention weights (unused currently — reserved).

        Returns:
            Phase-modulated hidden states [B, N, D].
        """
        if self.alpha < 1e-4:
            return hidden

        B, N, D = hidden.shape
        device  = hidden.device

        # Broadcast phases to [B, N]
        if phases.dim() == 1:
            ph = phases.unsqueeze(0).expand(B, -1)  # [B, N]
        else:
            ph = phases[:B]                          # [B, N]

        # Phase difference matrix: [B, N, N]
        # w_ij = (1 + cos(θᵢ − θⱼ)) / 2  ∈ [0, 1]
        ph_i = ph.unsqueeze(2)    # [B, N, 1]
        ph_j = ph.unsqueeze(1)    # [B, 1, N]
        phase_w = (1.0 + torch.cos(ph_i - ph_j)) / 2.0  # [B, N, N]

        # Normalise rows so total attention mass is preserved
        phase_w = phase_w / (phase_w.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        # Apply: each token's representation is a weighted sum over all tokens
        # weighted by phase coherence
        phase_modulated = torch.bmm(phase_w, hidden)  # [B, N, D]

        # Blend: α·phase_modulated + (1-α)·original
        alpha = self.alpha.to(device)
        return alpha * phase_modulated + (1.0 - alpha) * hidden

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, alpha={self.alpha.item():.3f}"
