"""
cgt/models/layer_v2.py
===========================
Isolated v2 layer stack: LayerNorm, FFN, Residual, HAKORNLayer, HAKORNEncoder.

KEY FIX: Residual connections
------------------------------
Legacy bug (layer.py lines 291, 298, 305, 308):
    hidden_states = residual + self.dropout(attn_output)   ← ILLEGAL on manifold

Direct tensor addition on the Lorentz hyperboloid violates the Minkowski
constraint.  The result of x + Δx does not lie on H^n.

Correct implementation (this file):
    HyperbolicResidualV2.forward(x, residual) uses spatial-only addition
    followed by time reconstruction:
        combined_spatial = x_spatial + residual_spatial
        x₀ = √(1/K + ‖combined_spatial‖²)
        result = cat([x₀, combined_spatial])
        result = substrate.proj(result)

This is equivalent to the legacy HyperbolicResidualHardened (which existed
but was NOT used by HAKORNLayer).  Here it is wired correctly.

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry import LorentzSubstrateV2
from cgt.models.hakorn_attention_v2 import HyperbolicKuramotoAttentionV2


def _sanitize(x: torch.Tensor, tag: str = "") -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian Layer Norm
# ─────────────────────────────────────────────────────────────────────────────

class RiemannianLayerNormV2(nn.Module):
    """
    Riemannian LayerNorm:
        1. log_map_zero(x)    → v  ∈ T_o  (tangent)
        2. LayerNorm(v_spatial)
        3. exp_map_zero(v_norm) → normalized manifold point

    γ is initialized > 1.0 to prevent radial collapse.
    Falls back to standard LayerNorm if substrate is None.
    """

    def __init__(
        self,
        d_model: int,
        substrate: Optional[LorentzSubstrateV2] = None,
        eps: float = 1e-5,
        gamma_init: float = 1.2,
        max_tangent_norm: float = 1.5,   # FIX: was implicit 4.0 (exp_map_zero default)
        # ROOT CAUSE OF rad=4.000 LOCK:
        # LayerNorm output in R^128 has L2 norm ≈ sqrt(128)*gamma ≈ 13.6,
        # which always saturates exp_map_zero's max_tangent_norm clamp.
        # With max_tangent_norm=4.0: enc_out radius is ALWAYS exactly 4.0,
        # making _radius_loss equivalent to optimizing a constant — zero effect.
        # With max_tangent_norm=1.5: output radius = 1.5, matching target_radius.
        # This makes radius_loss redundant (MSE=0 always), but crucially
        # prevents the logit_scale explosion that causes word salad:
        #   std(Minkowski inner) at r=1.5 = sinh²(1.5)/√128 ≈ 0.40  (vs 65.8 at r=4.0)
        #   → logit_scale stays near init; logit_std stabilizes below 1.2.
    ) -> None:
        super().__init__()
        self.d_model          = d_model
        self.substrate        = substrate
        self.eps              = eps
        self.max_tangent_norm = max_tangent_norm
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        if substrate is not None:
            with torch.no_grad():
                self.layer_norm.weight.fill_(gamma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.substrate is None:
            return self.layer_norm(x)

        x = _sanitize(x, "rln_in")
        orig_shape = x.shape
        orig_dtype = x.dtype
        ambient    = self.d_model + 1

        if x.shape[-1] == ambient:
            BL = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
            x_flat = x.reshape(BL, ambient)

            # log_map_zero returns float64 — keep it float64 through spatial ops
            v = self.substrate.log_map_zero(x_flat)   # [BL, n+1] float64
            v_spatial = v[:, 1:].float()              # [BL, n] cast to float32 for LayerNorm
            # (LayerNorm has float32 weight/bias — must match)
            v_spatial = _sanitize(v_spatial, "rln_v")

            v_norm = self.layer_norm(v_spatial)        # [BL, n] float32
            v_norm = v_norm.clamp(-5.0, 5.0)

            # Reconstruct tangent with zero time component — keep float32 for exp_map
            v_time = torch.zeros(BL, 1, device=x.device, dtype=torch.float32)
            v_full = torch.cat([v_time, v_norm], dim=-1)  # [BL, n+1] float32

            # FIX: pass max_tangent_norm=self.max_tangent_norm (default 1.5)
            # This replaces the old implicit default of 4.0 in exp_map_zero.
            # LayerNorm output norms in R^128 always exceed any reasonable
            # max_tangent_norm, so the clamp is always active → output radius
            # is deterministically set to max_tangent_norm here.
            x_out = self.substrate.exp_map_zero(v_full, max_tangent_norm=self.max_tangent_norm)
            x_out = self.substrate.proj(x_out)             # proj upcasts→float64→cast back
            return _sanitize(x_out.reshape(orig_shape), "rln_out")

        elif x.shape[-1] == self.d_model:
            return self.layer_norm(x)

        else:
            raise ValueError(
                f"RiemannianLayerNormV2: expected last dim {self.d_model} "
                f"or {ambient}, got {x.shape[-1]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic Residual  (THE KEY FIX)
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicResidualV2(nn.Module):
    """
    Correct residual connection for Lorentz manifold points.

    Algorithm
    ---------
    Input:
        x        : [*, n+1]  manifold point  (Lorentz ambient)
        residual : [*, n+1]  manifold point

    1. Extract spatial: x_s = x[..., 1:], r_s = residual[..., 1:]
    2. combined_s = x_s + r_s           (spatial addition — valid)
    3. clamp combined_s for safety
    4. Reconstruct time: t = √(1/K + ‖combined_s‖²)
    5. result = proj(cat([t, combined_s]))   (enforce manifold constraint)

    This is the correct Lorentz analogue of Euclidean x + Δx.
    It adds in the spatial subspace and re-derives the time component.

    Euclidean fallback (when substrate is None)
    -------------------------------------------
    Falls back to: x + residual  (standard transformer residual)
    Only use this when x is a pure Euclidean tensor.
    """

    def __init__(
        self,
        substrate: Optional[LorentzSubstrateV2],
        clamp: float = 10.0,
        max_spatial_norm: float = 5.0,
    ) -> None:
        super().__init__()
        self.substrate = substrate
        self.clamp_val = clamp
        self.max_spatial_norm = max_spatial_norm  # FIX: L2 norm ceiling per residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x        = _sanitize(x,        "res_x")
        residual = _sanitize(residual, "res_r")

        if self.substrate is None:
            return x + residual   # pure Euclidean path

        orig_dtype = x.dtype

        # Spatial-only addition (valid: spatial subspace is Euclidean)
        x_s = x[..., 1:]
        r_s = residual[..., 1:]
        combined_s = (x_s + r_s).clamp(-self.clamp_val, self.clamp_val)

        # FIX: clamp L2 norm to prevent radius accumulation across layers.
        # Per-element clamp(-10, 10) allows ||combined_s|| up to 10*sqrt(128)=113.
        # After 4 residuals, spatial norms compound: emb~2 + 4*layers ≈ 13.5.
        # Clamping the L2 norm (direction-preserving) bounds the geodesic radius
        # per layer without distorting the direction of the residual update.
        s_norm = combined_s.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        combined_s = torch.where(
            s_norm > self.max_spatial_norm,
            combined_s * (self.max_spatial_norm / s_norm),
            combined_s,
        )

        # FIX: reconstruct time component in float64 to avoid catastrophic cancellation.
        # In float32: t = sqrt(1/K + ||s||²) with ||s||~10 gives error ~1e-4 in
        # the Minkowski constraint because x0² and ||s||² nearly cancel.
        # In float64: same computation gives error ~1e-12.
        # proj() now also upcasts internally, so the final result is exact to f64,
        # then cast back to orig_dtype in a single step (~1e-7 error in float32).
        combined_s64 = combined_s.double()
        K64 = self.substrate.K.double().to(combined_s.device)
        t64 = torch.sqrt(
            (1.0 / K64 + (combined_s64 ** 2).sum(dim=-1, keepdim=True)).clamp(min=1e-15)
        )
        result = torch.cat([t64, combined_s64], dim=-1).to(orig_dtype)

        # proj() now upcasts to float64 internally → paranoid re-projection
        result = self.substrate.proj(result)
        return _sanitize(result, "res_out")


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicFFNV2(nn.Module):
    """
    FFN in tangent space: log_map_zero → linear → exp_map_zero.

    Pattern: M → T_o → FFN → T_o → M
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        substrate: Optional[LorentzSubstrateV2] = None,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.substrate = substrate

        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        acts = {"relu": F.relu, "gelu": F.gelu, "swish": lambda x: x * torch.sigmoid(x)}
        self.act = acts.get(activation, F.gelu)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize(x, "ffn_in")
        ambient = self.d_model + 1

        if self.substrate is not None and x.shape[-1] == ambient:
            orig_shape = x.shape
            orig_dtype = x.dtype
            BL = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
            x_flat = x.reshape(BL, ambient)

            # log_map_zero returns float64.
            # FIX: cast to float32 only for the linear layers (which have float32 weights).
            # Do NOT cast early — keep float64 precision where possible.
            v = self.substrate.log_map_zero(x_flat)   # [BL, n+1] float64
            v_s = v[:, 1:].to(orig_dtype)             # [BL, n] float32 (for fc1/fc2)

            # FFN in tangent space (Euclidean linear ops are valid on T_o)
            h = self.act(self.fc1(v_s))
            h = self.dropout(h)
            out_s = self.fc2(h)
            out_s = self.dropout(out_s)
            out_s = out_s.clamp(-10.0, 10.0)          # [BL, n] float32

            # Back to manifold: prepend zero time component
            # proj() upcasts to float64 internally → constraint exact to f64
            v_time = torch.zeros(BL, 1, device=x.device, dtype=orig_dtype)
            v_out  = torch.cat([v_time, out_s], dim=-1)   # [BL, n+1] float32
            x_out  = self.substrate.exp_map_zero(v_out)   # [BL, n+1] float32
            x_out  = self.substrate.proj(x_out)            # upcasts internally → ~1e-7 f32
            return _sanitize(x_out.reshape(orig_shape), "ffn_out")

        else:
            # Pure Euclidean path (substrate None or Euclidean input)
            h = self.act(self.fc1(x))
            h = self.dropout(h)
            out = self.fc2(h)
            return self.dropout(out)


# ─────────────────────────────────────────────────────────────────────────────
# Single HAKORN Layer
# ─────────────────────────────────────────────────────────────────────────────

class HAKORNLayerV2(nn.Module):
    """
    Single H-AKORN transformer layer (v2 — geometry correct).

    Fixes vs legacy layer.py
    -------------------------
    1. Residual connections use HyperbolicResidualV2 (spatial + time-reconstruct)
       instead of raw tensor addition.
    2. LayerNorm uses RiemannianLayerNormV2 (log → norm → exp).
    3. FFN uses HyperbolicFFNV2 (log → linear → exp).
    4. All intermediate tensors sanitized.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        substrate: Optional[LorentzSubstrateV2] = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        pre_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.pre_norm  = pre_norm
        self.substrate = substrate  # FIX: forward() references self.substrate

        self.attention = HyperbolicKuramotoAttentionV2(
            d_model             = d_model,
            num_heads           = num_heads,
            dropout             = dropout,
            substrate           = substrate,
            curvature           = curvature,
            coupling_strength   = coupling_strength,
            use_phase_modulation= use_phase_modulation,
        )
        self.feed_forward = HyperbolicFFNV2(
            d_model    = d_model,
            d_ff       = d_ff,
            dropout    = dropout,
            activation = activation,
            substrate  = substrate,
        )
        self.norm1 = RiemannianLayerNormV2(d_model, substrate, eps=layer_norm_eps)
        self.norm2 = RiemannianLayerNormV2(d_model, substrate, eps=layer_norm_eps)

        # Correct Riemannian residual — replaces raw `residual + delta`
        self.residual1 = HyperbolicResidualV2(substrate)
        self.residual2 = HyperbolicResidualV2(substrate)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns: (output, attn_weights, phases, order_param)
        """
        # ── Runtime geometry assertion (debug mode only) ──────────────────
        # Checks Minkowski constraint |<x,x>_L + 1/K| at layer input.
        # Disabled in production via torch.is_grad_enabled() check to avoid overhead.
        if self.substrate is not None and not torch.is_grad_enabled():
            K = self.substrate.K.item()
            h = hidden_states
            if h.shape[-1] == self.substrate.n + 1:
                mink = -h[..., 0:1] ** 2 + (h[..., 1:] ** 2).sum(-1, keepdim=True)
                err = (mink + 1.0 / K).abs().max().item()
                if err > 1e-2:   # soft warning threshold (not hard assert to avoid training crash)
                    import warnings
                    warnings.warn(
                        f"HAKORNLayerV2: Lorentz constraint violation at layer input: "
                        f"|<x,x>_L + 1/K| = {err:.2e} (threshold 1e-2). "
                        f"Expected < 1e-4 after geometry fixes.",
                        RuntimeWarning, stacklevel=2
                    )

        residual = hidden_states

        # ── Attention sub-layer ───────────────────────────────────────────
        if self.pre_norm:
            normed = self.norm1(hidden_states)
            attn_out, attn_w, phases, order = self.attention(
                normed, attention_mask, output_attentions
            )
            # Riemannian residual: residual ⊕ dropout(attn_out)
            hidden_states = self.residual1(self.dropout(attn_out), residual)
        else:
            attn_out, attn_w, phases, order = self.attention(
                hidden_states, attention_mask, output_attentions
            )
            hidden_states = self.norm1(
                self.residual1(self.dropout(attn_out), residual)
            )

        # ── FFN sub-layer ─────────────────────────────────────────────────
        residual = hidden_states
        if self.pre_norm:
            normed = self.norm2(hidden_states)
            ff_out = self.feed_forward(normed)
            hidden_states = self.residual2(self.dropout(ff_out), residual)
        else:
            ff_out = self.feed_forward(hidden_states)
            hidden_states = self.norm2(
                self.residual2(self.dropout(ff_out), residual)
            )

        return hidden_states, attn_w, phases, order

    def reset_phases(self) -> None:
        self.attention.reset_phases()


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Stack
# ─────────────────────────────────────────────────────────────────────────────

class HAKORNEncoderV2(nn.Module):
    """Stack of HAKORNLayerV2 blocks with optional final LayerNorm."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        substrate: Optional[LorentzSubstrateV2] = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        pre_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            HAKORNLayerV2(
                d_model             = d_model,
                num_heads           = num_heads,
                d_ff                = d_ff,
                dropout             = dropout,
                layer_norm_eps      = layer_norm_eps,
                substrate           = substrate,
                curvature           = curvature,
                coupling_strength   = coupling_strength,
                use_phase_modulation= use_phase_modulation,
                pre_norm            = pre_norm,
                activation          = activation,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = (
            RiemannianLayerNormV2(d_model, substrate, eps=layer_norm_eps)
            if pre_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[List], List, List]:
        all_hidden  = [] if output_hidden_states else None
        all_attns   = [] if output_attentions    else None
        all_phases  = []
        all_orders  = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(hidden_states)

            hidden_states, attn_w, phases, order = layer(
                hidden_states, attention_mask, output_attentions
            )
            all_phases.append(phases)
            all_orders.append(order)
            if output_attentions:
                all_attns.append(attn_w)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden.append(hidden_states)

        return hidden_states, all_attns, all_hidden, all_phases, all_orders

    def reset_phases(self) -> None:
        for layer in self.layers:
            layer.reset_phases()
