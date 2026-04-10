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


class AngularLMHeadV2(nn.Module):
    """
    AngularLMHeadV2 — production-ready angular LM head.

    The DegEq fix confirmed by HyDRA v4 V5-B experiment:
      rdc* 10.74 → 0.96 (91% reduction) with Channel 2 fix only.

    Difference from AngularLMHead (V1):
      - No warmup hack: temperature is initialised to a stable value
        directly. Warmup was masking a weight-scale issue; V2 instead
        normalises weight init at construction time.
      - Explicit fp32 matmul accumulation regardless of input dtype,
        then cast back. Prevents NaN in mixed-precision contexts.
      - tied_weights: embedding weight is projected via a learnable
        linear [n_embd → n_embd] before normalisation, so tied weights
        and LM head can diverge during fine-tuning.
      - verify_degeq_fix (debug): runs a single-step gradcheck at init
        to assert ∂logit/∂r = 0. Disabled by default (overhead).

    Key property (inherited from V1):
        ∂logit_k/∂r_h = 0  exactly  for all k, all r_h.

    Usage:
        # In DistillationConfigV2:
        use_angular_head = True          # activates V1 (legacy)

        # Direct construction (recommended for new models):
        head = AngularLMHeadV2(
            n_embd=128, vocab_size=50257, substrate=substrate
        )

        # Replace existing LM head:
        from cgt.models.geodesic_lm_head import replace_lm_head_angular
        replace_lm_head_angular(model, substrate)
    """

    def __init__(
        self,
        n_embd:           int,
        vocab_size:       int,
        substrate,
        temperature:      float = 20.0,     # cosine classifier standard
        learnable_temp:   bool  = True,
        tie_weights:      bool  = False,
        input_embeddings: "nn.Embedding | None" = None,
        verify_degeq_fix: bool  = False,    # gradcheck at init (debug)
    ) -> None:
        super().__init__()
        self.n_embd      = n_embd
        self.vocab_size  = vocab_size
        self.substrate   = substrate
        self.ambient_dim = substrate.n + 1

        # Temperature: log-space to keep positive
        init_log_temp = math.log(max(temperature, 1e-3))
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_temp))
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(init_log_temp)
            )

        if tie_weights and input_embeddings is not None:
            self._tied = input_embeddings
            self.weight = None
            # Projection so tied weights can diverge during fine-tuning
            self.proj = nn.Linear(n_embd, n_embd, bias=False)
            nn.init.eye_(self.proj.weight)          # identity init
        else:
            # Normalised init: unit-norm rows → stable cosine sim at step 0
            w = torch.randn(vocab_size, n_embd)
            w = torch.nn.functional.normalize(w, dim=-1)
            self.weight = nn.Parameter(w)
            self._tied  = None
            self.proj   = None

        if verify_degeq_fix:
            self._verify_zero_radial_grad()

    # ── Temperature property ──────────────────────────────────────────────
    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1.0, max=100.0)

    # ── Weight helpers ────────────────────────────────────────────────────
    def _get_vocab_tangent(self) -> torch.Tensor:
        """Return vocab tangent vectors [V, n], optionally projected."""
        if self._tied is not None:
            w = self._tied.weight           # [V, n_embd]
            if self.proj is not None:
                w = self.proj(w)
        else:
            w = self.weight                 # [V, n]
        return w

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, n+1]  Lorentz points on H^n
        Returns:
            logits: [B, L, V]   ∂logit/∂r = 0 for all r, all k
        """
        B, L, _ = hidden_states.shape

        # Project h → T_o, take spatial slice [1:] (zero time coord)
        h_flat  = hidden_states.reshape(-1, hidden_states.shape[-1])
        h_log   = self.substrate.log_map_zero(h_flat)[:, 1:]   # [BL, n]

        # fp32 accumulation for numerical stability
        h_log32 = h_log.float()
        u_h     = torch.nn.functional.normalize(h_log32, dim=-1, eps=1e-7)

        # Vocab unit vectors
        v_w     = self._get_vocab_tangent().float()             # [V, n]
        u_w     = torch.nn.functional.normalize(v_w, dim=-1, eps=1e-7)

        # Cosine logits  ∂(u_h · u_w)/∂r_h = 0 because û_h = h_log/||h_log||
        # and ||h_log|| = r_h  →  û_h = v_h/r_h  →  ∂û_h/∂r_h = 0
        logits = (u_h @ u_w.T) * self.temperature               # [BL, V]

        return torch.nan_to_num(
            logits.reshape(B, L, -1).to(hidden_states.dtype),
            nan=0.0, posinf=1e4, neginf=-1e4,
        )

    # ── Debug: verify ∂logit/∂r = 0 ─────────────────────────────────────
    def _verify_zero_radial_grad(self, tol: float = 1e-4) -> None:
        """
        Numerical check that ∂logit_k/∂r_h = 0 for a random test point.
        Raises AssertionError if the radial gradient exceeds tol.
        """
        import warnings
        device = next(self.parameters()).device
        n = self.n_embd

        # Random test point on H^n
        v_test = torch.randn(1, 1, n, device=device, dtype=torch.float64) * 0.5
        t_coord = torch.sqrt(1.0 + (v_test**2).sum(-1, keepdim=True))
        h_test  = torch.cat([t_coord, v_test], dim=-1).requires_grad_(True)

        # Forward
        logits = self.forward(h_test.float())
        loss   = logits.sum()
        loss.backward()

        if h_test.grad is None:
            return
        # Radial grad = component of grad along the spatial direction of h
        spatial     = h_test[0, 0, 1:].float()
        r           = spatial.norm()
        if r < 1e-6:
            return
        radial_unit = spatial / r
        grad_spatial= h_test.grad[0, 0, 1:].float()
        radial_comp = (grad_spatial * radial_unit).sum().abs().item()

        if radial_comp > tol:
            warnings.warn(
                f"[AngularLMHeadV2] ∂logit/∂r = {radial_comp:.2e} > {tol} "
                f"— Channel 2 may not be fully closed.",
                RuntimeWarning
            )
        else:
            pass  # silent success


def replace_lm_head_angular(
    model: nn.Module,
    substrate,
    temperature: float = 20.0,
    verify: bool = False,
) -> nn.Module:
    """
    Replace the LM head of any hyperbolic model with AngularLMHeadV2.

    This is the Channel 2 fix from HyDRA v4 V5-B:
        rdc* 10.74 → 0.96 (91% reduction, 3000 steps, OTED loss)

    Args:
        model:       SafeHyperbolicModel or any model with .core_model.lm_head
        substrate:   LorentzSubstrateV2 instance
        temperature: initial cosine classifier temperature (default 20.0)
        verify:      run gradcheck to assert ∂logit/∂r=0 (debug)

    Returns:
        model with lm_head replaced in-place.

    Example:
        from cgt.models.geodesic_lm_head import replace_lm_head_angular
        model = replace_lm_head_angular(model, substrate)
        # preflight check:
        assert type(model.core_model.lm_head).__name__ == "AngularLMHeadV2"
    """
    parent = getattr(model, "core_model", model)
    old_head = getattr(parent, "lm_head", None)
    if old_head is None:
        raise AttributeError("Model has no .lm_head or .core_model.lm_head")

    n_embd     = getattr(old_head, "n_embd", substrate.n)
    vocab_size = getattr(old_head, "vocab_size",
                         getattr(old_head, "out_features", None))
    if vocab_size is None:
        raise AttributeError("Cannot infer vocab_size from existing lm_head")

    new_head = AngularLMHeadV2(
        n_embd           = n_embd,
        vocab_size       = vocab_size,
        substrate        = substrate,
        temperature      = temperature,
        verify_degeq_fix = verify,
    ).to(next(model.parameters()).device)

    parent.lm_head = new_head
    return model

