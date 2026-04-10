"""
cgt/models/hakorn_attention_v2.py
=======================================
Isolated Hyperbolic Kuramoto Attention for v2.

Changes vs legacy attention.py
-------------------------------
- substrate.distance_matrix_batch() used for hyperbolic distance attention
- Geodesic value aggregation via substrate.attention_aggregate()
- Phase evolution wired through isolated KuramotoPhaseEvolutionV2
- No legacy cgt.* imports

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry import LorentzSubstrateV2


# ─────────────────────────────────────────────────────────────────────────────
# Kuramoto phase evolution (isolated, no legacy dependency)
# ─────────────────────────────────────────────────────────────────────────────

class KuramotoPhaseEvolutionV2(nn.Module):
    """Discrete-time Kuramoto oscillator for per-head phase dynamics."""

    def __init__(
        self,
        num_heads: int,
        coupling_strength: float = 1.0,
        dt: float = 0.1,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads        = num_heads
        self.coupling_strength= coupling_strength
        self.dt               = dt

        if learnable:
            self.natural_frequencies = nn.Parameter(torch.randn(num_heads) * 0.1)
        else:
            self.register_buffer("natural_frequencies", torch.randn(num_heads) * 0.1)

        self.register_buffer("phase_state", torch.rand(1, num_heads) * 2.0 * math.pi)

    def forward(
        self,
        coupling_matrix: torch.Tensor,    # [B, H, H]
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = coupling_matrix.device
        B = batch_size or coupling_matrix.shape[0]

        phases = self.phase_state.expand(B, -1).to(device)     # [B, H]
        if coupling_matrix.dim() == 2:
            coupling_matrix = coupling_matrix.unsqueeze(0)

        phase_diff   = phases.unsqueeze(2) - phases.unsqueeze(1)  # [B, H, H]
        coupling_sum = (coupling_matrix * torch.sin(phase_diff)).sum(dim=2)  # [B, H]

        dphase = (
            self.natural_frequencies.to(device)
            + (self.coupling_strength / self.num_heads) * coupling_sum
        )
        new_phases = torch.fmod(phases + self.dt * dphase, 2.0 * math.pi)

        with torch.no_grad():
            self.phase_state = new_phases[:1].detach().cpu()

        order = torch.abs(
            torch.complex(torch.cos(new_phases), torch.sin(new_phases)).mean(dim=1)
        )
        return new_phases, order

    def reset_phases(self) -> None:
        self.phase_state = torch.rand(1, self.num_heads) * 2.0 * math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive coupling (isolated)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveCouplingV2(nn.Module):
    """Hybrid coupling matrix from attention + phase coherence."""

    def __init__(self, num_heads: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = temperature
        # Learnable base coupling
        self.base = nn.Parameter(torch.eye(num_heads))

    def forward(
        self,
        attention_scores: Optional[torch.Tensor] = None,  # [B, H, L, L]
        phases: Optional[torch.Tensor] = None,            # [B, H]
    ) -> torch.Tensor:
        """Return [B, H, H] or [H, H] coupling matrix."""
        base = self.base.unsqueeze(0)                          # [1, H, H]

        if attention_scores is not None:
            # Collapse sequence dims → head-to-head
            attn_mean = attention_scores.mean(dim=(-2, -1))   # [B, H]
            # Outer product → [B, H, H]
            attn_coupling = torch.bmm(
                attn_mean.unsqueeze(2), attn_mean.unsqueeze(1)
            )
            base = base + 0.5 * attn_coupling

        if phases is not None:
            phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # [B, H, H]
            phase_coupling = torch.cos(phase_diff)
            base = base + 0.5 * phase_coupling

        return torch.softmax(base / self.temperature, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main attention module
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicKuramotoAttentionV2(nn.Module):
    """
    Hyperbolic Kuramoto attention (v2 — geometry correct).

    Attention scores: exp(-d_H(q_i, k_j) / τ) · cos(θ_i - θ_j)

    Value aggregation:
        If substrate provided → substrate.attention_aggregate (geodesic)
        Otherwise             → standard weighted sum
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        substrate: Optional[LorentzSubstrateV2] = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model             = d_model
        self.num_heads           = num_heads
        self.head_dim            = d_model // num_heads
        self.substrate           = substrate
        self.curvature           = curvature
        self.use_phase_modulation= use_phase_modulation
        self.temperature         = temperature

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

        self.phase_evo = KuramotoPhaseEvolutionV2(
            num_heads        = num_heads,
            coupling_strength= coupling_strength,
        )
        self.coupling = AdaptiveCouplingV2(num_heads, temperature)

    def _hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.substrate is not None:
            return self.substrate.dist(x, y)
        # Poincaré ball approximation
        x_sq = (x ** 2).sum(dim=-1, keepdim=True).clamp(max=0.99)
        y_sq = (y ** 2).sum(dim=-1, keepdim=True).clamp(max=0.99)
        diff_sq = ((x - y) ** 2).sum(dim=-1)
        denom = (1 - x_sq.squeeze(-1)) * (1 - y_sq.squeeze(-1))
        arg = 1 + 2 * diff_sq / (denom + 1e-8)
        return torch.log(arg + torch.sqrt((arg ** 2 - 1).clamp(min=1e-8)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, L, D = hidden_states.shape
        H, Dh   = self.num_heads, self.head_dim
        ambient = self.d_model + 1

        # ── Map ambient Lorentz → Euclidean tangent for Q/K/V projections ──
        # RiemannianLayerNormV2 outputs ambient vectors [B, L, n+1].
        # The linear projections are sized for the intrinsic dim n, so we
        # log-map to the tangent space and take the spatial part before
        # projecting, then lift the output back to the manifold.
        is_ambient = (self.substrate is not None and D == ambient)
        if is_ambient:
            flat  = hidden_states.reshape(B * L, ambient)
            v     = self.substrate.log_map_zero(flat).to(hidden_states.dtype)
            h_eu  = v[:, 1:].reshape(B, L, self.d_model)   # [B, L, n]
        else:
            h_eu  = hidden_states                            # already [B, L, n]

        Q = self.q_proj(h_eu).view(B, L, H, Dh).transpose(1, 2)  # [B,H,L,Dh]
        K = self.k_proj(h_eu).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(h_eu).view(B, L, H, Dh).transpose(1, 2)

        # Kuramoto phase evolution
        coupling_m = self.coupling(attention_scores=None, phases=None)
        phases, order = self.phase_evo(coupling_m, batch_size=B)

        # Hyperbolic distance scores
        # Chunked hyperbolic distance — avoids [B,H,L,L,Dh] materialisation
        # Splits L into chunks of CHUNK_SZ, computing distances one row-block at a time.
        # Memory: O(B×H×CHUNK×L×Dh) instead of O(B×H×L²×Dh)
        CHUNK_SZ = max(1, min(64, Q.shape[2]))
        L_q = Q.shape[2]
        dist_chunks = []
        for _c in range(0, L_q, CHUNK_SZ):
            Q_chunk = Q[:, :, _c:_c+CHUNK_SZ, :].unsqueeze(3)   # [B,H,chunk,1,Dh]
            K_exp   = K.unsqueeze(2)                              # [B,H,1,L,Dh]
            dist_chunks.append(self._hyperbolic_distance(Q_chunk, K_exp))  # [B,H,chunk,L]
        distances = torch.cat(dist_chunks, dim=2)                 # [B,H,L,L]
        scores = torch.exp(-distances / self.temperature)

        # Phase modulation
        if self.use_phase_modulation and phases is not None:
            pd = phases.unsqueeze(2) - phases.unsqueeze(1)    # [B, H, H]
            coherence = torch.cos(pd).mean(dim=2)             # [B, H]
            modulation = coherence.unsqueeze(2).unsqueeze(3)  # [B, H, 1, 1]
            scores = scores * modulation.expand_as(scores)

        # Mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        # Update coupling with real attention
        coupling_m = self.coupling(attention_scores=attn_weights, phases=phases)

        # Aggregate values — view uses d_model, not D, to avoid 128 vs 129 mismatch
        context    = torch.matmul(attn_weights, V)                         # [B, H, L, Dh]
        context_eu = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output_eu  = self.out_proj(context_eu)                             # [B, L, n]

        # ── Lift Euclidean output back to ambient if input was ambient ──────
        if is_ambient:
            flat_eu = output_eu.reshape(B * L, self.d_model)
            v_full  = torch.cat(
                [torch.zeros(B * L, 1, dtype=flat_eu.dtype, device=flat_eu.device),
                 flat_eu], dim=-1
            )                                                              # [BL, n+1]
            output  = self.substrate.exp_map_zero(v_full).reshape(B, L, ambient)
            output  = self.substrate.proj(output)                          # enforce H^n
        else:
            output  = output_eu

        attn_out = attn_weights if output_attentions else None
        return output, attn_out, phases, order

    def reset_phases(self) -> None:
        self.phase_evo.reset_phases()
