"""
cgt/models/lm_head_v2.py  — PATCHED v5: Symmetric Hyperbolic Vocab Embeddings
===============================================================================
PATCH HISTORY
-------------
v4: proj() on vocab weights (eliminates x0 inconsistency, but does NOT bound radius)
v5: exp_map_zero + max_tangent_norm on vocab (bounds geodesic radius structurally)

ROOT CAUSE OF DegEq (confirmed across Variants A and D):
---------------------------------------------------------
Hidden states:   geodesic radius ≤ sinh(r_max=1.5) via RiemannianLayerNorm
Vocab weights:   unbounded — norm grows as log(t) under implicit margin maximization

This asymmetry allows logit_std to grow monotonically without bound, even when
vocab is on the manifold (v4/Variant D). The optimizer shifts weight to growing
the spatial norm rather than improving angular discrimination.

FIX (Variant E): Symmetric tangent-space parameterization
----------------------------------------------------------
Store vocab as tangent vectors w ∈ R^n (not ambient R^(n+1)).
Lift to manifold via exp_map_zero with max_tangent_norm=r_max_vocab.

This imposes:
    ||w_i||_geodesic ≤ r_max_vocab  for all vocab embeddings w_i

And therefore:
    logit_std ≤ logit_scale * sinh(r_max_vocab) / sqrt(n)

With r_max_vocab=3.0, logit_scale_max=2.5 (reduced from 5.0 in v5.1):
    logit_std_max ≈ 2.5 * sinh(3.0) / sqrt(128) ≈ 2.2

With r_max_vocab=3.0, logit_scale=3.5 (typical):
    logit_std_ceiling ≈ 3.1  (above healthy range but bounded)

CRITICAL: r_max_vocab ≠ r_max_hidden
    r_max_hidden = 1.5  (encoder, via RLN)
    r_max_vocab  = 3.0  (decoder, needs more range for token discrimination)

Using 1.5 for vocab would create UNDERCONFIDENT regime (logit_std < 1.0).
The asymmetry in r_max is intentional: encoder learns geometry, decoder
learns discrimination at appropriate scale.

DegEq ELIMINATION:
    With r_max_vocab=3.0, logit_std has a hard ceiling.
    RDC proxy = logit_std / (l_hidden + eps) cannot diverge because
    numerator is bounded. DegEq (rdc_ema > 15) requires
    logit_std >> l_hidden; with a ceiling this cannot persist.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn

from cgt.geometry import LorentzSubstrateV2
from cgt.dynamics.riemannian_update_v2 import ensure_lorentz_v2


def _sanitize(x: torch.Tensor, name: str = "") -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class HyperbolicLMHeadV2(nn.Module):
    """
    LM head using intrinsic Lorentz geometry (Minkowski inner product).

    v5: vocab embeddings stored as tangent vectors, lifted to manifold via
    exp_map_zero with bounded radius. Eliminates DegEq structurally by
    imposing a geodesic radius ceiling on vocabulary representations.
    """

    # Vocab radius — larger than hidden (1.5) to allow sufficient logit contrast
    VOCAB_R_MAX: float = 3.0

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        substrate: LorentzSubstrateV2,
        tie_weights: bool = True,
        input_embeddings: nn.Embedding | None = None,
        normalize: bool = False,
        eps: float = 1e-8,
        learnable_scale: bool = True,
        init_logit_scale: float = 5.0,
        vocab_r_max: float = 3.0,   # geodesic radius bound for vocab embeddings
    ) -> None:
        super().__init__()
        self.n_embd      = n_embd
        self.vocab_size  = vocab_size
        self.substrate   = substrate
        self.tie_weights = tie_weights
        self.normalize   = normalize
        self.eps         = eps
        self.ambient_dim = substrate.n + 1
        self.vocab_r_max = vocab_r_max   # v5: configurable radius bound

        # Temperature
        init_temp = 1.0 / math.sqrt(max(n_embd, 1))
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(init_temp))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_temp))
        self.learnable_scale = learnable_scale

        if tie_weights:
            if input_embeddings is None:
                raise ValueError(
                    "HyperbolicLMHeadV2: tie_weights=True requires input_embeddings."
                )
            self._tied_embeddings = input_embeddings
            self.weight = None
            emb_dim = input_embeddings.weight.shape[1]
            if emb_dim != self.ambient_dim:
                self._lift = nn.Linear(emb_dim, self.ambient_dim, bias=False)
                nn.init.normal_(self._lift.weight, std=0.01)
            else:
                self._lift = None
        else:
            # v5: store as TANGENT vectors [V, n] — NOT ambient [V, n+1]
            # Optimizer works in flat tangent space; exp_map_zero lifts to manifold
            self.weight = nn.Parameter(
                torch.randn(vocab_size, n_embd) * 0.01   # small init, tangent space
            )
            self._tied_embeddings = None
            self._lift = None

    def _get_weight(self) -> torch.Tensor:
        """
        Return vocab embeddings on the Lorentz hyperboloid, bounded by vocab_r_max.

        v5 approach:
            self.weight stores tangent vectors [V, n]
            exp_map_zero lifts to manifold with max_tangent_norm=vocab_r_max
            Result: geodesic radius of each vocab embedding ≤ vocab_r_max

        Gradient flow: autodiff through exp_map_zero to self.weight (tangent space).
        The optimizer works in R^n (well-behaved), the manifold constraint is
        enforced at inference time via exp_map_zero.

        For tied weights: apply same constraint to input embeddings.
        """
        if self.tie_weights and self._tied_embeddings is not None:
            w = self._tied_embeddings.weight   # [V, emb_dim]
            if self._lift is not None:
                w = self._lift(w)              # [V, ambient]
                # Ambient → spatial for exp_map
                w_spatial = w[..., 1:]         # [V, n]
            else:
                w_spatial = w                  # [V, n] tangent
            # Prepend zero time component: [V, n] → [V, n+1]
            zeros = torch.zeros(*w_spatial.shape[:-1], 1,
                                device=w_spatial.device, dtype=w_spatial.dtype)
            v_tangent = torch.cat([zeros, w_spatial], dim=-1)
            return self.substrate.exp_map_zero(v_tangent, max_tangent_norm=self.vocab_r_max)

        # v5: self.weight is [V, n] pure spatial tangent.
        # exp_map_zero expects [..., n+1] with v[..., 0] = 0 (origin tangent).
        # Prepend zero time component: [V, n] → [V, n+1].
        w = self.weight
        zeros = torch.zeros(*w.shape[:-1], 1, device=w.device, dtype=w.dtype)
        v_tangent = torch.cat([zeros, w], dim=-1)   # [V, n+1], v[:,0]=0
        return self.substrate.exp_map_zero(
            v_tangent,
            max_tangent_norm=self.vocab_r_max
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, n+1] Lorentz points
        Returns:
            logits: [B, L, V]
        """
        hidden_states = _sanitize(hidden_states, "lm_input")

        # 1. Enforce manifold constraint on hidden states
        x = ensure_lorentz_v2(hidden_states, self.substrate, debug_tag="lm_head")

        # 2. Get vocab embeddings — bounded on hyperboloid (v5)
        w = _sanitize(self._get_weight(), "lm_weight")   # [V, n+1]

        # 3. Minkowski inner product in float64
        x64 = x.double()
        w64 = w.double()

        time_inner  = torch.einsum("blt,vt->blv", x64[..., :1], w64[:, :1])
        space_inner = torch.einsum("bls,vs->blv", x64[..., 1:], w64[:, 1:])
        inner = space_inner - time_inner

        # 4. Temperature scaling
        tau = self.logit_scale.clamp(0.01, 2.5).double()   # v5.1: reduced ceiling to prevent OVERCONFIDENT drift
        logits = inner * tau

        # 5. Cast back, sanitize
        logits = logits.to(hidden_states.dtype)
        return _sanitize(logits, "lm_logits")
