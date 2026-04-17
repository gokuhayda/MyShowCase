"""
cgt/models/angular_physics_layer.py
====================================
HyDRA v9 — HAKORNLayerV2_v9: drop-in replacement for HAKORNLayerV2 that
inserts the AngularPhysicsGate between attention and the on-manifold
residual.

Backward compatibility
----------------------
- Subclasses HAKORNLayerV2; all v7 behaviour is preserved when the gate
  is not attached or when the feature flag is off.
- Does NOT modify any v7 module in place.
- Can be swapped into HAKORNEncoderV2 by replacing the layer class.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cgt.models.layer_v2 import HAKORNLayerV2
from cgt.models.angular_physics import AngularPhysicsGate, AngularPhysicsConfig


class HAKORNLayerV2_v9(HAKORNLayerV2):
    """HAKORN layer with angular-physics gating on the attention residual.

    Forward pass differences vs v7:

        residual = x
        normed   = norm1(x)
        h_attn, attn_w, ... = attention(normed, mask, output_attentions=True)
        x_spatial = normed[..., 1:]     # pre-attention spatial features
        gate, cos_align, K = angular_gate(x_spatial, attn_w)
        h_attn = apply_gate(h_attn, gate, ambient=True)   # spatial-only scale
        x = residual1(h_attn, residual)                   # unchanged on-manifold residual

    Everything after the attention residual (FFN sub-layer etc.) is
    inherited unchanged from v7.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        substrate=None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        pre_norm: bool = True,
        activation: str = "gelu",
        # v9 additions
        angular_gate_enabled: bool = True,
        angular_gate_config: Optional[AngularPhysicsConfig] = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            substrate=substrate,
            curvature=curvature,
            coupling_strength=coupling_strength,
            use_phase_modulation=use_phase_modulation,
            pre_norm=pre_norm,
            activation=activation,
        )
        self.angular_gate_enabled = angular_gate_enabled
        if angular_gate_enabled:
            self.angular_gate = AngularPhysicsGate(
                angular_gate_config or AngularPhysicsConfig()
            )
        else:
            self.angular_gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        # ── Fast path: feature flag off → exact v7 behaviour ─────────────
        if not self.angular_gate_enabled or self.angular_gate is None:
            return super().forward(hidden_states, attention_mask, output_attentions)

        residual = hidden_states

        # ── Attention sub-layer with mandatory attn-weight return ────────
        if self.pre_norm:
            normed = self.norm1(hidden_states)
            attn_out, attn_w, phases, order = self.attention(
                normed, attention_mask, output_attentions=True
            )
            # Spatial component of the pre-attention normed features
            x_spatial = normed[..., 1:] if normed.shape[-1] == self.d_model + 1 else normed
            gate, cos_align, K_norm = self.angular_gate(x_spatial, attn_w)

            # Gate the attention output — spatial-only if ambient Lorentz
            ambient = (attn_out.shape[-1] == self.d_model + 1)
            attn_out_gated = self.angular_gate.apply_gate(attn_out, gate, ambient=ambient)

            # On-manifold residual (unchanged v7 machinery)
            hidden_states = self.residual1(self.dropout(attn_out_gated), residual)
        else:
            attn_out, attn_w, phases, order = self.attention(
                hidden_states, attention_mask, output_attentions=True
            )
            x_spatial = hidden_states[..., 1:] if hidden_states.shape[-1] == self.d_model + 1 else hidden_states
            gate, cos_align, K_norm = self.angular_gate(x_spatial, attn_w)
            ambient = (attn_out.shape[-1] == self.d_model + 1)
            attn_out_gated = self.angular_gate.apply_gate(attn_out, gate, ambient=ambient)
            hidden_states = self.norm1(
                self.residual1(self.dropout(attn_out_gated), residual)
            )

        # ── FFN sub-layer (unchanged v7) ─────────────────────────────────
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

        # Respect caller's output_attentions flag for the returned attn_w
        returned_attn_w = attn_w if output_attentions else None
        return hidden_states, returned_attn_w, phases, order


def upgrade_transformer_to_v9(model, angular_gate_config: Optional[AngularPhysicsConfig] = None):
    """In-place upgrade: replace every HAKORNLayerV2 in model.encoder.layers
    with a HAKORNLayerV2_v9, reusing the existing weights of attention/FFN/
    norms/residuals. Only the new AngularPhysicsGate parameters are fresh.

    Returns the modified model (same object).
    """
    from cgt.models.transformer_v2 import HyperbolicTransformerV2  # lazy
    assert isinstance(model, HyperbolicTransformerV2), type(model)
    cfg = model.config

    new_layers = nn.ModuleList()
    for old in model.encoder.layers:
        v9 = HAKORNLayerV2_v9(
            d_model              = cfg.n_embd,
            num_heads            = cfg.n_head,
            d_ff                 = cfg.ffn_dim,
            dropout              = cfg.dropout,
            layer_norm_eps       = cfg.layer_norm_eps,
            substrate            = model.substrate,
            curvature            = -cfg.initial_curvature,
            coupling_strength    = 1.0,
            use_phase_modulation = True,
            pre_norm             = True,
            activation           = "gelu",
            angular_gate_enabled = True,
            angular_gate_config  = angular_gate_config,
        )
        # Transfer existing weights — v9 is a true superset of v7
        v9.attention    = old.attention
        v9.feed_forward = old.feed_forward
        v9.norm1        = old.norm1
        v9.norm2        = old.norm2
        v9.residual1    = old.residual1
        v9.residual2    = old.residual2
        v9.dropout      = old.dropout
        new_layers.append(v9)

    model.encoder.layers = new_layers
    return model
