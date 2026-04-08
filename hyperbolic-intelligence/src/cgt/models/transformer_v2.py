"""
cgt/models/transformer_v2.py
=================================
HyperbolicTransformerV2 — fully isolated, geometry-correct transformer.

All fixes applied:
    ✅  LM head: log_map_zero (not slice [..., 1:])
    ✅  Residuals: HyperbolicResidualV2 (not raw tensor +)
    ✅  LayerNorm: log → norm → exp (not Euclidean)
    ✅  Geometry: float64 in all hyperbolic ops
    ✅  Projection: proj() enforced after every manifold update
    ✅  No legacy cgt.* imports

Config precedence rule applied in HyperbolicTransformerConfigV2.__post_init__.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry import LorentzConfigV2, LorentzSubstrateV2
from cgt.models.lm_head_v2 import HyperbolicLMHeadV2
from cgt.models.layer_v2 import HAKORNEncoderV2, RiemannianLayerNormV2


def _sanitize(x: torch.Tensor, tag: str = "") -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperbolicTransformerConfigV2:
    """
    Config for HyperbolicTransformerV2.

    Config Precedence Rule (all aliased fields in __post_init__)
    -------------------------------------------------------------
        if canonical_value != default:
            use canonical_value
        elif alias_value is not None:
            use alias_value
        else:
            use default

    Canonical fields
    ----------------
    vocab_size, n_embd, n_layer, n_head, n_positions, ffn_ratio,
    dropout, attention_dropout, initial_curvature, learnable_curvature,
    radius_max, attention_temperature, tie_word_embeddings

    Legacy aliases
    --------------
    hidden_size  → n_embd
    num_layers   → n_layer
    num_heads    → n_head
    max_seq_len  → n_positions
    """

    # ── canonical ──────────────────────────────────────────────────────────
    vocab_size: int              = 50257
    n_embd: int                  = 64
    n_layer: int                 = 4
    n_head: int                  = 4
    n_positions: int             = 512
    ffn_ratio: int               = 4
    dropout: float               = 0.0
    attention_dropout: float     = 0.0
    initial_curvature: float     = 1.0
    learnable_curvature: bool    = False
    curvature_min: float         = 0.1
    curvature_max: float         = 10.0
    radius_max: float            = 5.0
    attention_temperature: float = 1.0
    tie_word_embeddings: bool    = True
    embedding_init_std: float    = 0.001
    layer_norm_eps: float        = 1e-5

    # ── legacy aliases ─────────────────────────────────────────────────────
    hidden_size: Optional[int]   = None   # alias → n_embd
    num_layers: Optional[int]    = None   # alias → n_layer
    num_heads: Optional[int]     = None   # alias → n_head
    max_seq_len: Optional[int]   = None   # alias → n_positions

    # ── private sentinels ──────────────────────────────────────────────────
    _D_N_EMBD: int       = field(default=64,  init=False, repr=False)
    _D_N_LAYER: int      = field(default=4,   init=False, repr=False)
    _D_N_HEAD: int       = field(default=4,   init=False, repr=False)
    _D_N_POS: int        = field(default=512, init=False, repr=False)

    def __post_init__(self) -> None:
        # n_embd
        if self.n_embd != self._D_N_EMBD:
            pass
        elif self.hidden_size is not None:
            self.n_embd = self.hidden_size

        # n_layer
        if self.n_layer != self._D_N_LAYER:
            pass
        elif self.num_layers is not None:
            self.n_layer = self.num_layers

        # n_head
        if self.n_head != self._D_N_HEAD:
            pass
        elif self.num_heads is not None:
            self.n_head = self.num_heads

        # n_positions
        if self.n_positions != self._D_N_POS:
            pass
        elif self.max_seq_len is not None:
            self.n_positions = self.max_seq_len

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self) -> int:
        return self.n_embd * self.ffn_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Embedding layer
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicEmbeddingV2(nn.Module):
    """
    Token + position embeddings lifted to Lorentz manifold.

    Embeddings are stored as Euclidean spatial vectors,
    projected on-the-fly via exp_map_zero:
        v = [0, e_spatial]   → exp_map_zero(v)  → H^n point
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_positions: int,
        substrate: LorentzSubstrateV2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        init_std: float = 0.001,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.substrate = substrate

        self.token_embeddings = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_positions, d_model)
        self.register_buffer(
            "position_ids", torch.arange(max_positions).unsqueeze(0)
        )
        self.drop      = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Small init — keep embeddings near origin for manifold stability
        nn.init.normal_(self.token_embeddings.weight, std=init_std)
        nn.init.normal_(self.position_embeddings.weight, std=init_std)
        if pad_token_id is not None:
            with torch.no_grad():
                self.token_embeddings.weight[pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embeddings(input_ids)               # [B, L, d]
        if position_ids is None:
            position_ids = self.position_ids[:, :L].to(device)
        pos_emb = self.position_embeddings(position_ids)         # [B, L, d]

        e_spatial = tok_emb + pos_emb
        e_spatial = self.layer_norm(e_spatial)
        e_spatial = self.drop(e_spatial)

        # Clamp before lifting (prevent huge norms from blowing up exp_map)
        e_norm  = e_spatial.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = 2.0
        scale   = torch.where(e_norm > max_norm, max_norm / e_norm, torch.ones_like(e_norm))
        e_spatial = e_spatial * scale
        e_spatial = _sanitize(e_spatial, "emb_spatial")

        # Lift: [B, L, d] → [B, L, d+1]  via exp_map_zero
        BL = B * L
        e_flat = e_spatial.reshape(BL, self.d_model)
        v_time = torch.zeros(BL, 1, device=device, dtype=e_flat.dtype)
        v = torch.cat([v_time, e_flat], dim=-1)                  # [BL, d+1]
        x = self.substrate.exp_map_zero(v)                       # [BL, d+1]
        x = self.substrate.proj(x)                               # hard projection
        x = _sanitize(x, "emb_out")
        return x.reshape(B, L, self.d_model + 1)                 # [B, L, d+1]


# ─────────────────────────────────────────────────────────────────────────────
# Full transformer
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicTransformerV2(nn.Module):
    """
    Fully isolated, geometry-correct hyperbolic transformer (v2).

    All violations from the legacy codebase are fixed here.
    No imports from cgt.* legacy namespace.
    """

    def __init__(self, config: Optional[HyperbolicTransformerConfigV2] = None) -> None:
        super().__init__()
        self.config = config or HyperbolicTransformerConfigV2()
        cfg = self.config

        # ── Substrate (geometry engine) ───────────────────────────────────
        lorentz_cfg = LorentzConfigV2(
            intrinsic_dim      = cfg.n_embd,
            learnable_curvature= cfg.learnable_curvature,
            initial_curvature  = cfg.initial_curvature,
            curvature_min      = cfg.curvature_min,
            curvature_max      = cfg.curvature_max,
        )
        self.substrate = LorentzSubstrateV2(lorentz_cfg)

        # ── Embeddings ────────────────────────────────────────────────────
        self.embeddings = HyperbolicEmbeddingV2(
            vocab_size  = cfg.vocab_size,
            d_model     = cfg.n_embd,
            max_positions= cfg.n_positions,
            substrate   = self.substrate,
            dropout     = cfg.dropout,
            init_std    = cfg.embedding_init_std,
        )

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = HAKORNEncoderV2(
            num_layers          = cfg.n_layer,
            d_model             = cfg.n_embd,
            num_heads           = cfg.n_head,
            d_ff                = cfg.ffn_dim,
            dropout             = cfg.dropout,
            layer_norm_eps      = cfg.layer_norm_eps,
            substrate           = self.substrate,
            curvature           = -cfg.initial_curvature,
            coupling_strength   = 1.0,
            use_phase_modulation= True,
        )

        # ── LM head (THE FIX: log_map_zero, not [..., 1:]) ────────────────
        self.lm_head = HyperbolicLMHeadV2(
            n_embd      = cfg.n_embd,
            vocab_size  = cfg.vocab_size,
            substrate   = self.substrate,
            tie_weights = cfg.tie_word_embeddings,
            input_embeddings = (
                self.embeddings.token_embeddings
                if cfg.tie_word_embeddings else None
            ),
        )

    # ── Attention mask ────────────────────────────────────────────────────

    def _make_causal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        device = input_ids.device
        causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        return causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict:
        # Embeddings → manifold points [B, L, n+1]
        hidden_states = self.embeddings(input_ids)

        # Build attention mask
        if attention_mask is None:
            attention_mask = self._make_causal_mask(input_ids)
        elif attention_mask.dim() == 2:
            B, L = attention_mask.shape
            attention_mask = (
                attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, L, L)
            )

        # Encoder
        enc_out, all_attns, all_hidden, all_phases, all_orders = self.encoder(
            hidden_states, attention_mask, output_attentions, output_hidden_states
        )

        # LM head (log_map_zero internally — no illegal slice)
        logits = self.lm_head(enc_out)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict:
            return {
                "logits":            logits,
                "loss":              loss,
                "hidden_states":     enc_out,
                "all_hidden_states": all_hidden,
                "all_attentions":    all_attns,
                "all_phases":        all_phases,
                "all_order_params":  all_orders,
            }
        return (logits, loss, enc_out)

    # ── Generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            out    = self.forward(generated, return_dict=True)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                kth = torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None:
                sorted_l, sorted_idx = torch.sort(logits, descending=True)
                cum_prob = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove   = cum_prob > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0]  = False
                sorted_l = sorted_l.masked_fill(remove, float("-inf"))
                # FIX: scatter into -inf base (not zeros) to avoid unmasked zero-logit tokens
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_l)

            if do_sample:
                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated

    # ── Diagnostics ───────────────────────────────────────────────────────

    def manifold_fidelity(self) -> Dict[str, float]:
        """Check mean |<x,x>_L + 1/K| across sampled embeddings."""
        with torch.no_grad():
            device = self.embeddings.token_embeddings.weight.device
            n = min(32, self.config.vocab_size)
            ids = torch.randint(0, self.config.vocab_size, (n,), device=device)
            # Build dummy input to get manifold points
            ids_2d = ids.unsqueeze(0)  # [1, n]
            x = self.embeddings(ids_2d).reshape(n, -1)   # [n, d+1]
            v = self.substrate.manifold_violation(x).item()
            upper = self.substrate.check_upper_sheet(x)
            return {
                "mean_violation": v,
                "upper_sheet_ok": upper,
            }

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embeddings.token_embeddings.weight.numel()
            n -= self.embeddings.position_embeddings.weight.numel()
        return n
