"""
cgt/entrypoint.py
======================
SafeHyperbolicModel — the single clean entrypoint for the v2 system.

Usage
-----
    from cgt.entrypoint import SafeHyperbolicModel, SafeModelConfig

    model = SafeHyperbolicModel(SafeModelConfig(vocab_size=50257, n_embd=64))
    out   = model(input_ids)            # → logits [B, L, V]
    gen   = model.generate(input_ids)   # → generated token IDs

This class:
    - Uses ONLY cgt.* modules (zero legacy cgt.* imports)
    - Bypasses the legacy pipeline entirely
    - Wires DomainGuard and ParanoidMonitor
    - Exposes geometry diagnostics
    - Accepts legacy alias fields via config precedence rule

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from cgt.models.transformer_v2 import (
    HyperbolicTransformerConfigV2,
    HyperbolicTransformerV2,
)
from cgt.integration.dynamic_slm_v2 import DynamicSLMWrapperV2
from cgt.config.dynamics_config_v2 import DynamicsConfigV2
from cgt.guard.domain_guard_v2 import DomainGuardV2, GUARD_V2
from cgt.guard.paranoid_monitor_v2 import ParanoidMonitorV2, MONITOR_V2


# ─────────────────────────────────────────────────────────────────────────────
# SafeModelConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafeModelConfig:
    """
    Unified config for SafeHyperbolicModel.

    Config Precedence Rule applied in __post_init__ for ALL aliased fields:
        if canonical_value != default:
            use canonical_value
        elif alias_value is not None:
            use alias_value
        else:
            use default

    Canonical transformer fields
    ----------------------------
    vocab_size, n_embd, n_layer, n_head, n_positions, ffn_ratio, dropout,
    initial_curvature, learnable_curvature, tie_word_embeddings

    Canonical dynamics fields
    -------------------------
    use_dynamics, hyperbolic_dim, dt, num_steps, coupling_strength,
    noise_std, learnable_frequencies

    Legacy aliases
    --------------
    hidden_size  → n_embd
    num_layers   → n_layer
    num_heads    → n_head
    max_seq_len  → n_positions
    hyp_dim      → hyperbolic_dim
    n_ctx        → n_positions (sequence length)

    Safe mode
    ---------
    enable_paranoid_monitor : install torch monkey-patch to intercept
                              any tensor addition on manifold tensors.
                              True by default in debug mode.
    """

    # ── transformer canonical ─────────────────────────────────────────────
    vocab_size: int              = 50257
    n_embd: Optional[int]        = None   # None → resolved in __post_init__ (default 128)
    n_layer: int                 = 4
    n_head: int                  = 4
    n_positions: int             = 512
    ffn_ratio: int               = 4
    dropout: float               = 0.0
    initial_curvature: float     = 1.0
    learnable_curvature: bool    = False
    tie_word_embeddings: bool    = True
    embedding_init_std: float    = 0.001
    layer_norm_eps: float        = 1e-5

    # ── dynamics canonical ────────────────────────────────────────────────
    use_dynamics: bool           = True
    hyperbolic_dim: int          = 128
    dt: float                    = 0.02
    num_steps: int               = 5
    coupling_strength: float     = 0.1
    noise_std: float             = 0.0
    learnable_frequencies: bool  = True

    # ── guard ─────────────────────────────────────────────────────────────
    enable_paranoid_monitor: bool = False   # set True for debug/dev
    paranoid_debug: bool          = False

    # ── legacy aliases ─────────────────────────────────────────────────────
    hidden_size: Optional[int]   = None   # → n_embd
    num_layers: Optional[int]    = None   # → n_layer
    num_heads: Optional[int]     = None   # → n_head
    max_seq_len: Optional[int]   = None   # → n_positions
    hyp_dim: Optional[int]       = None   # → hyperbolic_dim
    n_ctx: Optional[int]         = None   # → n_positions

    # ── sentinels ─────────────────────────────────────────────────────────
    # n_embd uses Optional[int]=None — no sentinel needed; None means "not set"
    _D_N_LAYER: int  = field(default=4,   init=False, repr=False)
    _D_N_HEAD: int   = field(default=4,   init=False, repr=False)
    _D_N_POS: int    = field(default=512, init=False, repr=False)
    _D_HYP: int      = field(default=128, init=False, repr=False)

    def __post_init__(self) -> None:
        """Apply config precedence rule to all aliased fields."""

        # n_embd: explicit int wins → alias hidden_size → default 128
        # Using Optional[int]=None avoids sentinel collision when user passes n_embd=128.
        if self.n_embd is not None:
            pass  # explicit value wins (including n_embd=128 passed by user)
        elif self.hidden_size is not None:
            self.n_embd = self.hidden_size
        else:
            self.n_embd = 128  # true default

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

        # n_positions — canonical wins, then n_ctx alias, then max_seq_len
        if self.n_positions != self._D_N_POS:
            pass
        elif self.n_ctx is not None:
            self.n_positions = self.n_ctx
        elif self.max_seq_len is not None:
            self.n_positions = self.max_seq_len

        # hyperbolic_dim
        if self.hyperbolic_dim != self._D_HYP:
            pass
        elif self.hyp_dim is not None:
            self.hyperbolic_dim = self.hyp_dim


# ─────────────────────────────────────────────────────────────────────────────
# SafeHyperbolicModel
# ─────────────────────────────────────────────────────────────────────────────

class SafeHyperbolicModel(nn.Module):
    """
    Safe, isolated entrypoint for the v2 hyperbolic language model.

    Coexistence guarantee
    ---------------------
    This class imports NOTHING from the legacy cgt.* namespace.
    It can be imported alongside legacy code without interference.
    The guard singletons (GUARD_V2, MONITOR_V2) are v2-private.

    Wiring
    ------
    core_model      : HyperbolicTransformerV2   (geometry-correct transformer)
    slm_wrapper     : DynamicSLMWrapperV2       (Kuramoto dynamics, Riemannian update)
    guard           : DomainGuardV2             (assertion layer)
    monitor         : ParanoidMonitorV2         (optional torch monkey-patch)
    """

    def __init__(
        self,
        config: Optional[SafeModelConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        cfg = config or SafeModelConfig(**kwargs)
        self.cfg = cfg

        # ── Patch 3: Precision guards (HyperRad / HSTD v7.0 standard) ────
        # Disable TF32: its reduced mantissa causes cumulative drift in
        # Minkowski inner products during Kuramoto integration, pushing
        # safe_acosh inputs into the unstable zone and generating NaN grads.
        # NOTE: do NOT call torch.set_default_dtype(float64) globally — that
        # would make nn.LayerNorm weights float64 while geodesic code explicitly
        # casts v_spatial to float32 before LayerNorm, causing a dtype mismatch.
        # float64 precision is already applied locally inside each geodesic op
        # (log_map_zero, exp_map_zero, proj) via internal .double() casts.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Guard Kuramoto integration parameters
        if cfg.use_dynamics:
            assert getattr(cfg, "dt", 0.05) <= 0.05, (
                f"Precision guard: dt > 0.05 induces unrecoverable tangent deviation "
                f"(got dt={cfg.dt})"
            )
            assert getattr(cfg, "num_steps", 10) >= 1, (
                f"Precision guard: num_steps must be >= 1 for valid trajectory "
                f"(got num_steps={cfg.num_steps})"
            )

        # ── Build transformer config ──────────────────────────────────────
        t_cfg = HyperbolicTransformerConfigV2(
            vocab_size          = cfg.vocab_size,
            n_embd              = cfg.n_embd,
            n_layer             = cfg.n_layer,
            n_head              = cfg.n_head,
            n_positions         = cfg.n_positions,
            ffn_ratio           = cfg.ffn_ratio,
            dropout             = cfg.dropout,
            initial_curvature   = cfg.initial_curvature,
            learnable_curvature = cfg.learnable_curvature,
            tie_word_embeddings = cfg.tie_word_embeddings,
            embedding_init_std  = cfg.embedding_init_std,
            layer_norm_eps      = cfg.layer_norm_eps,
        )

        # ── Core model (HyperbolicTransformerV2) ──────────────────────────
        self.core_model = HyperbolicTransformerV2(t_cfg)

        # ── Dynamic SLM wrapper (optional, driven by use_dynamics) ────────
        if cfg.use_dynamics:
            d_cfg = DynamicsConfigV2(
                num_oscillators      = cfg.n_positions,
                embed_dim            = cfg.n_embd,
                hyperbolic_dim       = cfg.hyperbolic_dim,
                dt                   = cfg.dt,
                num_steps            = cfg.num_steps,
                coupling_strength    = cfg.coupling_strength,
                noise_std            = cfg.noise_std,
                use_hyperbolic_coupling = False,
                use_dynamics         = True,
                record_trajectory    = False,
                learnable_frequencies= cfg.learnable_frequencies,
            )
            self.slm_wrapper: Optional[DynamicSLMWrapperV2] = DynamicSLMWrapperV2(
                slm1=None, slm2=None, config=d_cfg
            )
        else:
            self.slm_wrapper = None

        # ── Guards ────────────────────────────────────────────────────────
        self.guard: DomainGuardV2 = GUARD_V2
        self.monitor: ParanoidMonitorV2 = ParanoidMonitorV2(debug=cfg.paranoid_debug)

        if cfg.enable_paranoid_monitor:
            self.monitor.install()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict:
        """
        Full forward pass through v2 system.

        Pipeline:
            input_ids
                → HyperbolicTransformerV2  (embeddings + encoder + LM head)
                → [optional] DynamicSLMWrapperV2 on encoder hidden states
                → logits [B, L, V]
        """
        out = self.core_model(
            input_ids           = input_ids,
            attention_mask      = attention_mask,
            labels              = labels,
            output_attentions   = output_attentions,
            output_hidden_states= output_hidden_states,
            return_dict         = True,
        )

        # Optionally route encoder hidden states through Riemannian dynamics
        if self.slm_wrapper is not None and self.cfg.use_dynamics:
            h = out["hidden_states"]   # [B, L, n+1]
            h_evolved = self.slm_wrapper(h)
            # Re-run LM head with evolved hidden states
            logits_evolved = self.core_model.lm_head(h_evolved)
            out = dict(out)
            out["logits"]         = logits_evolved
            out["hidden_states"]  = h_evolved

        return out

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
        return self.core_model.generate(
            input_ids       = input_ids,
            max_new_tokens  = max_new_tokens,
            temperature     = temperature,
            top_k           = top_k,
            top_p           = top_p,
            do_sample       = do_sample,
            eos_token_id    = eos_token_id,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def geometry_report(self) -> Dict:
        """Return geometry health metrics."""
        fidelity = self.core_model.manifold_fidelity()
        report = {
            "manifold_violation_mean": fidelity["mean_violation"],
            "upper_sheet_ok":          fidelity["upper_sheet_ok"],
            "curvature_K":             self.core_model.substrate.K.item(),
            "paranoid_monitor_active": self.monitor.installed,
        }
        if self.slm_wrapper and self.slm_wrapper.last_trajectory:
            traj = self.slm_wrapper.last_trajectory
            report["phase_entropy"]       = traj.phase_entropy()
            report["order_parameter"]     = traj.final_order_parameter()
        return report

    def enable_zero_trust(self, debug: bool = False) -> None:
        """Install paranoid monitor to intercept any manifold tensor addition."""
        if not self.monitor.installed:
            self.monitor.debug = debug
            self.monitor.install()

    def disable_zero_trust(self) -> None:
        self.monitor.uninstall()

    def num_parameters(self, non_embedding: bool = True) -> int:
        return self.core_model.num_parameters(non_embedding)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"SafeHyperbolicModel("
            f"vocab={self.cfg.vocab_size}, "
            f"n_embd={self.cfg.n_embd}, "
            f"n_layer={self.cfg.n_layer}, "
            f"n_head={self.cfg.n_head}, "
            f"params={n:,}, "
            f"dynamics={'on' if self.cfg.use_dynamics else 'off'}"
            f")"
        )
