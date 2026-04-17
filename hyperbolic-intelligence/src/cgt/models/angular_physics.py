"""
cgt/models/angular_physics.py
==============================
HyDRA v9 — Angular Physics Gate.

Purpose
-------
Replace the HyDRA v8.1 structural inversion (physics ANTI-aligned with
semantics, cos_align ≈ -0.40) with an angular coupling signal that is
*monotonically aligned with semantic similarity by construction*.

Design rules (non-negotiable)
-----------------------------
1. The gate MODULATES MAGNITUDE, not attention routing. Attention logits
   are never touched — QK^T/√d goes through softmax unchanged.
2. The kernel uses ANGULAR similarity (cosine of spatial component),
   not geodesic distance. This fixes the inversion in the HLLM paper:
     CE ↓ semantic distance  ⇒  cosine(x_i, x_j) ↑  ⇒  K ↑
   which is the opposite of what geodesic coupling did.
3. All alignment computations detach the attention weights. The gate is
   the ONLY trainable path through which physics influences the residual.
4. Centering the kernel (K - mean(K)) is required — without it, a Softplus/
   positive kernel trivially produces cos_align > 0 regardless of actual
   directional agreement. This follows HLLM §3.5 (PGR).

Integration contract
--------------------
Call site in HAKORNLayerV2_v9.forward (see v9_layer.py or notebook):

    h_attn, attn_w, phases, order = self.attention(
        normed, attention_mask, output_attentions=True
    )
    gate = self.angular_gate(x_prenorm, attn_w)   # [B, L], in (0, 1)
    h_attn_gated = self.angular_gate.apply_gate(h_attn, gate)
    hidden_states = self.residual1(h_attn_gated, residual)

This keeps HyperbolicResidualV2 untouched and composes cleanly with
the existing on-manifold residual machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AngularPhysicsConfig:
    """Config for AngularPhysicsGate.

    bias_init : float
        Initial bias of the gate linear. Default 0.0 → sigmoid(0) = 0.5,
        i.e. the gate starts neutral (same effective magnitude as v7).
    eps : float
        Numerical epsilon for norms and std normalizations.
    detach_attn_for_alignment : bool
        If True (default), attn weights feeding cos_align are detached,
        so cos_align is a *diagnostic* signal that flows gradients ONLY
        through the kernel K and the gate linear. This is the strict
        reading of the v9 spec ("gate modulates magnitude, not routing").
    """
    bias_init: float = 0.0
    eps: float = 1e-6
    detach_attn_for_alignment: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Angular Physics Gate
# ─────────────────────────────────────────────────────────────────────────────

class AngularPhysicsGate(nn.Module):
    """Angular coupling gate for HyDRA v9.

    Forward pipeline:

        x_unit    = x_spatial / ||x_spatial||                 (unit sphere)
        K_raw     = x_unit @ x_unit.T                          [B, L, L]   cos sim
        K_norm    = (K_raw - mean(K_raw)) / (std(K_raw) + eps)              centred+scaled
        A         = attn.mean(dim=heads).detach()              [B, L, L]
        cos_align = (A * K_norm).sum(dim=-1)                   [B, L]
        gate      = sigmoid(W * cos_align + b)                 [B, L]

    The gate is then applied multiplicatively to the attention output
    via :meth:`apply_gate` (which handles both ambient-Lorentz and
    Euclidean tensor shapes).

    Diagnostics collected per forward:
        - last_cos_align  : [B, L]      alignment signal
        - last_gate       : [B, L]      sigmoid output
        - last_corr_attn_K: scalar      Pearson(attn, K_norm) over last batch

    These are stored as detached tensors for logging; they do not
    participate in any backward pass initiated by the caller.
    """

    def __init__(self, config: Optional[AngularPhysicsConfig] = None) -> None:
        super().__init__()
        self.config = config or AngularPhysicsConfig()
        # Scalar linear on the 1-D cos_align per token.
        # Weight init = 1.0 so gate≈sigmoid(cos_align) at step 0.
        self.W = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.full((1,), float(self.config.bias_init)))

        # Diagnostic buffers (detached, overwritten every forward)
        self.register_buffer("last_cos_align", torch.zeros(1), persistent=False)
        self.register_buffer("last_gate",      torch.zeros(1), persistent=False)
        self.register_buffer("last_corr_attn_K", torch.zeros(1), persistent=False)

    # ── Kernel construction ──────────────────────────────────────────────

    @staticmethod
    def _spatial(x: torch.Tensor) -> torch.Tensor:
        """Extract the Euclidean/spatial component of a (possibly ambient
        Lorentz) tensor.

        If x has last-dim d+1 and looks like a Lorentz point, strip the
        time coordinate. Otherwise return x unchanged.
        """
        # Heuristic: only strip when called explicitly with ambient input.
        # The caller (HAKORNLayerV2_v9) is responsible for passing the right
        # tensor; we offer this helper but do NOT auto-detect.
        return x

    def angular_kernel(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """Compute centred, scaled angular kernel K_norm ∈ R[B, L, L].

        Args
        ----
        x_spatial : [B, L, d]  — spatial / Euclidean features

        Returns
        -------
        K_norm : [B, L, L]
        """
        eps = self.config.eps
        # Unit-normalize on last dim (keep gradients through x_unit)
        x_unit = x_spatial / x_spatial.norm(dim=-1, keepdim=True).clamp(min=eps)
        K = torch.matmul(x_unit, x_unit.transpose(-1, -2))           # [B, L, L] ∈ [-1, 1]

        # Per-batch centering & scaling. We .detach() the std to prevent
        # the gradient from collapsing the scale (the std of K is a
        # trivially minimisable quantity if left trainable).
        K_mean = K.mean(dim=(-1, -2), keepdim=True)
        K_cent = K - K_mean
        K_std  = K_cent.detach().std(dim=(-1, -2), keepdim=True).clamp(min=eps)
        K_norm = K_cent / K_std
        return K_norm

    # ── Alignment signal ─────────────────────────────────────────────────

    def compute_cos_align(
        self,
        attn: torch.Tensor,
        K_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Centred cosine alignment between attention distribution and
        the normalized angular kernel.

        Args
        ----
        attn   : [B, H, L, L] or [B, L, L]    softmax output
        K_norm : [B, L, L]

        Returns
        -------
        cos_align : [B, L]     per-token alignment score
        """
        if attn.dim() == 4:
            A = attn.mean(dim=1)                     # [B, L, L]
        else:
            A = attn
        if self.config.detach_attn_for_alignment:
            A = A.detach()
        # Element-wise product and sum over the "key" axis
        cos_align = (A * K_norm).sum(dim=-1)          # [B, L]
        return cos_align

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x_spatial: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the per-token gate.

        Args
        ----
        x_spatial : [B, L, d]     pre-attention spatial features
        attn      : [B, H, L, L]  softmax weights (detached inside)

        Returns
        -------
        gate      : [B, L]        in (0, 1)
        cos_align : [B, L]        diagnostic (attached to gate graph via W·)
        K_norm    : [B, L, L]     diagnostic
        """
        K_norm    = self.angular_kernel(x_spatial)
        cos_align = self.compute_cos_align(attn, K_norm)            # [B, L]
        # Scalar W, scalar b → elementwise affine on cos_align
        gate_logits = self.W * cos_align + self.b                   # [B, L]
        gate        = torch.sigmoid(gate_logits)                    # [B, L]

        # Update diagnostics (detached, no graph retention)
        with torch.no_grad():
            self.last_cos_align = cos_align.detach().float()
            self.last_gate      = gate.detach().float()
            # Pearson correlation between attn (mean over heads) and K_norm,
            # flattened over (B, L, L). Fully detached.
            if attn.dim() == 4:
                A_det = attn.mean(dim=1).detach().float()
            else:
                A_det = attn.detach().float()
            K_det = K_norm.detach().float()
            a = A_det.reshape(-1)
            k = K_det.reshape(-1)
            a = a - a.mean()
            k = k - k.mean()
            denom = (a.norm() * k.norm()).clamp(min=self.config.eps)
            self.last_corr_attn_K = (a * k).sum() / denom

        return gate, cos_align, K_norm

    # ── Gate application ─────────────────────────────────────────────────

    @staticmethod
    def apply_gate(
        h_attn: torch.Tensor,
        gate: torch.Tensor,
        ambient: bool = False,
    ) -> torch.Tensor:
        """Multiply the attention output by the per-token gate.

        For ambient Lorentz tensors (last-dim n+1) we scale only the
        spatial component and leave the time coordinate to be reconstructed
        downstream by HyperbolicResidualV2 (which does exactly this).

        Args
        ----
        h_attn  : [B, L, D]         attention output
        gate    : [B, L]            in (0, 1)
        ambient : bool              if True, D = n+1 and we gate spatial only

        Returns
        -------
        h_attn_gated : [B, L, D]
        """
        g = gate.unsqueeze(-1)                                       # [B, L, 1]
        if ambient:
            # [B, L, 1+n]: leave time as-is, scale spatial
            t = h_attn[..., :1]
            s = h_attn[..., 1:]
            s = s * g
            return torch.cat([t, s], dim=-1)
        return h_attn * g


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run directly: `python angular_physics.py`)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 16, 32
    x     = torch.randn(B, L, d, requires_grad=True)
    # Fake softmax attn
    raw   = torch.randn(B, H, L, L)
    attn  = torch.softmax(raw, dim=-1)

    gate_mod = AngularPhysicsGate()
    print(f"trainable params: {sum(p.numel() for p in gate_mod.parameters())}")

    gate, cos_align, K_norm = gate_mod(x, attn)
    print(f"gate      : {tuple(gate.shape)}  mean={gate.mean().item():.4f}  std={gate.std().item():.4f}")
    print(f"cos_align : {tuple(cos_align.shape)}  mean={cos_align.mean().item():+.4f}")
    print(f"K_norm    : {tuple(K_norm.shape)}  mean={K_norm.mean().item():+.2e}  std={K_norm.std().item():.4f}")
    print(f"corr(attn,K): {gate_mod.last_corr_attn_K.item():+.4f}")

    # Gradient sanity: gate params and x should both receive gradients
    loss = (gate - 0.7).pow(2).mean()
    loss.backward()
    assert gate_mod.W.grad is not None and gate_mod.W.grad.abs().item() > 0, "W has no grad"
    assert gate_mod.b.grad is not None, "b has no grad"
    assert x.grad is not None and x.grad.abs().sum().item() > 0, "x has no grad"
    print("gradients flow: W", gate_mod.W.grad.item(), " b", gate_mod.b.grad.item())
    print("✓ self-test passed")
