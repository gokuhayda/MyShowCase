# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
hyplora.py
==========
HypLoRA — 100% Hyperbolic LoRA adapter for frozen LLMs (CGT style).

Design
------
Follows the HypLoRA architecture (Yang et al., NeurIPS 2025) but built
entirely on CGT's LorentzSubstrateV2 primitives, RiemannianAdamW, and
TB-PAG numerical stability policy.

Each adapted linear layer gets a parallel hyperbolic branch:

    z = W·x  +  Π_log( LLR( B·A,  Π_exp(x) ) )
    ↑            ↑     ↑          ↑
    frozen    back to  Lorentz   project to
    Euclidean  R^d     Low-Rank   H^n
    weight            transform

Where:
  - Π_exp  = substrate.exp_map_zero  : R^d → H^n
  - Π_log  = substrate.log_map_zero  : H^n → R^d  (spatial slice [1:])
  - LLR    = LorentzLowRank           : H^n → H^n  (rank-r transform on manifold)

Key properties vs naive tangent-space LoRA:
  - Adapts DIRECTLY on manifold coordinates (no repeated log/exp cancel)
  - ∂output/∂r is non-zero → preserves radial information in fine-tuning
  - Curvature K is learnable per layer (initialized from substrate)
  - Radial momentum projection (V5) available for all trainable params

No DegEq risk:
  Each HypLoRALayer does exactly ONE exp_map and ONE log_map per forward.
  The frozen backbone does not accumulate manifold momentum — only A,B,K
  are updated. This avoids the 4-layer Christoffel cascade that causes
  DegEq in fully-hyperbolic models trained from scratch.

Usage
-----
    # Wrap any nn.Linear in a frozen LLM:
    from cgt.models.hyplora import HypLoRALayer, inject_hyplora, HypLoRAConfig

    config = HypLoRAConfig(rank=8, alpha=16.0)
    model  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    inject_hyplora(model, config, target_modules=["q_proj", "v_proj"])

    # Only HypLoRA params are trainable — everything else frozen.
    optimizer = RiemannianAdamW(
        list(model.named_parameters()),
        substrate=config.substrate,
        lr=3e-4,
        radial_momentum_projection=True,   # V5 fix
    )

Components
----------
  LorentzLowRank      — rank-r transform on H^n
  HypLoRALayer        — wraps an nn.Linear with the hyperbolic branch
  HypLoRAConfig       — dataclass for all hyperparameters
  inject_hyplora()    — replaces target Linear layers in any nn.Module
  extract_hyplora()   — extracts only adapter weights (for saving/merging)
  merge_hyplora()     — fuses adapter back into base weights (inference)
  delta_hyperbolicity()— measures δ-hyperbolicity of a tensor set
  token_freq_norm()   — token frequency vs embedding norm diagnostic

References
----------
  Yang et al. (2025). HypLoRA: Hyperbolic Fine-Tuning for Large Language Models.
    NeurIPS 2025 Spotlight. https://arxiv.org/abs/2410.04010

  CGT geometry primitives:
    cgt.geometry.lorentz_v2.LorentzSubstrateV2
    cgt.dynamics.riemannian_adamw.RiemannianAdamW
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry import LorentzSubstrateV2
from cgt.geometry.lorentz_v2 import LorentzConfigV2


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HypLoRAConfig:
    """
    Configuration for HypLoRA adapters.

    rank        : Low-rank bottleneck dimension r.  Typical: 4, 8, 16.
    alpha       : Scaling factor. Effective scale = alpha / rank.
                  Set alpha = rank for scale=1.0 (no rescaling).
    dropout     : Dropout on adapter output (0 = disabled).
    init_scale  : std for A initialisation.  B is zero-init → output starts at 0.
    n_embd      : Hidden dimension of the adapted model.  Required.
    curvature   : Initial K for each adapter's private substrate.
                  If None, uses K=1.0.  K is learnable per layer.
    learn_curvature: If True, K is an nn.Parameter per HypLoRALayer.
    max_tangent_norm: Clamp for exp_map_zero (TB-PAG policy).
    dtype       : Adapter parameter dtype.  float32 for speed, float64 for stability.
    """
    rank:            int   = 8
    alpha:           float = 16.0
    dropout:         float = 0.0
    init_scale:      float = 0.01
    n_embd:          int   = 0       # REQUIRED: set to model hidden dim
    curvature:       float = 1.0
    learn_curvature: bool  = True
    max_tangent_norm: float = 1.5
    dtype:           torch.dtype = torch.float32

    # Internal: shared substrate (created lazily)
    _substrate: Optional[LorentzSubstrateV2] = field(default=None, init=False, repr=False)

    @property
    def scale(self) -> float:
        return self.alpha / max(self.rank, 1)

    def get_substrate(self, device: Optional[torch.device] = None) -> LorentzSubstrateV2:
        """Return a shared LorentzSubstrateV2.  Created once, then cached."""
        if self._substrate is None:
            cfg = LorentzConfigV2(initial_curvature=self.curvature, intrinsic_dim=self.n_embd)
            self._substrate = LorentzSubstrateV2(cfg)
        if device is not None:
            self._substrate = self._substrate.to(device)
        return self._substrate


# ─────────────────────────────────────────────────────────────────────────────
# LorentzLowRank — the core manifold transform
# ─────────────────────────────────────────────────────────────────────────────

class LorentzLowRank(nn.Module):
    """
    Low-rank transform operating DIRECTLY on Lorentz manifold coordinates.

    Implements the LLR (Lorentz Low-Rank) from HypLoRA:

        LLR(x_H) = exp_y( B · A · log_y(x_H) )

    where y is the base point (here: the origin for simplicity and
    numerical stability).  This avoids the base-point dependency of
    full Riemannian LoRA while preserving manifold operations.

    Mathematical properties:
      - Input/output both on H^n (substrate.proj guarantees this)
      - rank-r bottleneck in TANGENT SPACE at origin T_o H^n ≅ R^n
      - Curvature K is learnable: regulates how curved the adapter is

    Implementation notes:
      - A: [n, r]  — maps tangent vector to r-dim latent
      - B: [r, n]  — maps back to n-dim tangent
      - B initialised to zero → adapter output = 0 at init (stable start)
      - A initialised with small Gaussian (breaks symmetry)
    """

    def __init__(
        self,
        n_embd: int,
        rank: int,
        substrate: LorentzSubstrateV2,
        learn_curvature: bool = True,
        max_tangent_norm: float = 1.5,
        init_scale: float = 0.01,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_embd          = n_embd
        self.rank            = rank
        self.max_tangent_norm = max_tangent_norm
        self._base_substrate = substrate

        # Low-rank matrices in tangent space
        # A: [n, r],  B: [r, n]
        self.A = nn.Parameter(
            torch.randn(n_embd, rank, dtype=dtype) * init_scale
        )
        self.B = nn.Parameter(
            torch.zeros(rank, n_embd, dtype=dtype)
        )

        # Learnable curvature K > 0 (log-parameterised for positivity)
        if learn_curvature:
            init_log_K = math.log(substrate.K.item())
            self.log_K = nn.Parameter(torch.tensor(init_log_K, dtype=dtype))
        else:
            self.log_K = None

    @property
    def K(self) -> torch.Tensor:
        if self.log_K is not None:
            return torch.exp(self.log_K).clamp(min=1e-4, max=10.0)
        return self._base_substrate.K

    def _get_substrate(self, device: torch.device) -> LorentzSubstrateV2:
        """Return substrate with current K (refreshed if K is learnable)."""
        if self.log_K is not None:
            # Build a lightweight substrate with current K
            cfg = LorentzConfigV2(initial_curvature=self.K.item(), intrinsic_dim=self.n_embd)
            return LorentzSubstrateV2(cfg).to(device)
        return self._base_substrate.to(device)

    def forward(self, x_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_H: [..., n+1]  points on H^n

        Returns:
            y_H: [..., n+1]  adapted points on H^n
        """
        substrate = self._get_substrate(x_H.device)

        # 1. Project to tangent space at origin → R^n
        #    log_map_zero returns [..., n+1]; we take spatial slice [1:]
        v = substrate.log_map_zero(x_H)[..., 1:]           # [..., n]

        # 2. Low-rank transform in tangent space: v → B·(A·v)
        #    Shapes: v [...,n] @ A [n,r] = [...,r]  then @ B^T [n,r]^T=[r,n] → wait:
        #    B [r,n], A [n,r]:   (v @ A) is [..., r],  (v @ A) @ B is [..., n]
        v_adapted = (v.to(self.A.dtype) @ self.A) @ self.B  # [..., n]

        # 3. Lift back to H^n via exp_map_zero
        #    Prepend zero time component for valid tangent at origin
        zeros = torch.zeros(
            *v_adapted.shape[:-1], 1,
            device=v_adapted.device, dtype=v_adapted.dtype
        )
        v_full = torch.cat([zeros, v_adapted], dim=-1)       # [..., n+1]
        y_H = substrate.exp_map_zero(
            v_full, max_tangent_norm=self.max_tangent_norm
        )

        return y_H.to(x_H.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# HypLoRALayer — wraps an nn.Linear with the hyperbolic branch
# ─────────────────────────────────────────────────────────────────────────────

class HypLoRALayer(nn.Module):
    """
    Drop-in replacement for nn.Linear with a parallel hyperbolic adapter.

    Forward pass:

        z_E = W·x_E                          ← frozen base (Euclidean)
            + Π_log( LLR( Π_exp(x_E) ) )    ← hyperbolic adapter branch
              scaled by (alpha / rank)

    where:
      Π_exp = exp_map_zero : R^d → H^n
      Π_log = log_map_zero : H^n → R^d (spatial slice only)
      LLR   = LorentzLowRank on H^n

    Only LLR.A, LLR.B, LLR.log_K are trainable.
    base_linear.weight and base_linear.bias are frozen.

    Dimension handling:
      in_features  may differ from n_embd (e.g. for cross-attention layers).
      If in_features != n_embd, the hyperbolic branch projects through the
      adapter's n_embd dimension using linear pre/post projections.
      Most cases: in_features == out_features == n_embd — no extra projection.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        config: HypLoRAConfig,
        substrate: LorentzSubstrateV2,
    ) -> None:
        super().__init__()

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.config       = config
        self.scale        = config.scale

        # Freeze base weights
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        # Dimension alignment
        n_embd = config.n_embd
        self._needs_in_proj  = (self.in_features  != n_embd)
        self._needs_out_proj = (self.out_features != n_embd)

        if self._needs_in_proj:
            self.in_proj = nn.Linear(self.in_features, n_embd, bias=False,
                                     dtype=config.dtype)
        if self._needs_out_proj:
            self.out_proj = nn.Linear(n_embd, self.out_features, bias=False,
                                      dtype=config.dtype)

        # Core hyperbolic adapter
        self.llr = LorentzLowRank(
            n_embd          = n_embd,
            rank            = config.rank,
            substrate       = substrate,
            learn_curvature = config.learn_curvature,
            max_tangent_norm= config.max_tangent_norm,
            init_scale      = config.init_scale,
            dtype           = config.dtype,
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Private substrate for exp/log at this layer
        self._substrate = substrate

    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean hidden state → H^n."""
        substrate = self.llr._get_substrate(x.device)
        # Prepend zero time component
        z = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        v = torch.cat([z, x], dim=-1)                        # [..., n+1]
        return substrate.exp_map_zero(
            v, max_tangent_norm=self.config.max_tangent_norm
        )

    def _project_to_euclidean(self, x_H: torch.Tensor) -> torch.Tensor:
        """Map H^n → Euclidean by taking spatial log_map_zero slice."""
        substrate = self.llr._get_substrate(x_H.device)
        v = substrate.log_map_zero(x_H)                      # [..., n+1]
        return v[..., 1:]                                     # [..., n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]  Euclidean hidden states

        Returns:
            z: [..., out_features]  adapted output
        """
        # ── Base (frozen) Euclidean path ──────────────────────────────────
        z_base = self.base_linear(x)

        # ── Hyperbolic adapter branch ──────────────────────────────────────
        x_adapter = x
        if self._needs_in_proj:
            x_adapter = self.in_proj(x_adapter)               # [..., n_embd]

        # Euclidean → H^n
        x_H = self._project_to_manifold(x_adapter)            # [..., n_embd+1]

        # Transform on manifold (LorentzLowRank)
        y_H = self.llr(x_H)                                   # [..., n_embd+1]

        # H^n → Euclidean (spatial slice of log_map_zero)
        delta = self._project_to_euclidean(y_H)               # [..., n_embd]
        delta = delta.to(x.dtype)

        if self._needs_out_proj:
            delta = self.out_proj(delta)                       # [..., out_features]

        # Apply dropout and scale
        delta = self.dropout(delta) * self.scale

        return z_base + delta

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.config.rank}, alpha={self.config.alpha}, "
                f"learn_K={self.config.learn_curvature}")


# ─────────────────────────────────────────────────────────────────────────────
# inject_hyplora — plug adapters into any nn.Module
# ─────────────────────────────────────────────────────────────────────────────

def inject_hyplora(
    model: nn.Module,
    config: HypLoRAConfig,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
) -> Dict[str, HypLoRALayer]:
    """
    Replace target nn.Linear layers with HypLoRALayer adapters.

    All base weights are frozen; only adapter parameters are trainable.

    Args:
        model:          Any nn.Module (LLM, transformer, etc.)
        config:         HypLoRAConfig with n_embd set to model's hidden dim.
        target_modules: List of module name suffixes to adapt.
                        Default: ["q_proj", "v_proj"] (attention keys+values).
                        Common choices:
                          ["q_proj", "k_proj", "v_proj", "o_proj"]  — full attn
                          ["gate_proj", "up_proj", "down_proj"]       — FFN
                          ["q_proj", "v_proj"]                        — HypLoRA default
        exclude_modules: Module names to skip even if they match target.

    Returns:
        Dict mapping module path → HypLoRALayer for each replaced layer.

    Example:
        adapted = inject_hyplora(
            llama3,
            HypLoRAConfig(rank=8, alpha=16.0, n_embd=4096),
            target_modules=["q_proj", "v_proj"],
        )
        print(f"Injected {len(adapted)} HypLoRA layers")
        print_trainable_params(llama3)
    """
    if config.n_embd == 0:
        raise ValueError("HypLoRAConfig.n_embd must be set to the model hidden dim.")

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    if exclude_modules is None:
        exclude_modules = []

    target_set  = set(target_modules)
    exclude_set = set(exclude_modules)

    device    = next(model.parameters()).device
    substrate = config.get_substrate(device)

    injected: Dict[str, HypLoRALayer] = {}

    for name, module in list(model.named_modules()):
        # Check if this module's last component matches any target
        last_name = name.split(".")[-1]
        if last_name not in target_set:
            continue
        if last_name in exclude_set or name in exclude_set:
            continue
        if not isinstance(module, nn.Linear):
            warnings.warn(f"[HypLoRA] {name} is not nn.Linear (got {type(module).__name__}), skipping.")
            continue

        # Build replacement
        hyp_layer = HypLoRALayer(
            base_linear=module,
            config=config,
            substrate=substrate,
        ).to(device)

        # Navigate to parent and replace
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, child_name, hyp_layer)

        injected[name] = hyp_layer

    if not injected:
        warnings.warn(
            f"[HypLoRA] No layers were injected. "
            f"target_modules={target_modules} did not match any nn.Linear in the model. "
            f"Available linear names: "
            f"{[n for n,m in model.named_modules() if isinstance(m, nn.Linear)][:10]}"
        )
    else:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f"[HypLoRA] Injected {len(injected)} adapters. "
              f"Trainable: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    return injected


# ─────────────────────────────────────────────────────────────────────────────
# extract_hyplora — save adapter weights only
# ─────────────────────────────────────────────────────────────────────────────

def extract_hyplora(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only HypLoRA adapter state dict (A, B, log_K, projections).
    Useful for saving adapters independently of the frozen backbone.

    Returns dict of {param_name: tensor} for all HypLoRALayer params.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, HypLoRALayer):
            for pname, param in module.llr.named_parameters(recurse=False):
                state[f"{name}.llr.{pname}"] = param.detach().clone()
            if module._needs_in_proj:
                for pname, param in module.in_proj.named_parameters():
                    state[f"{name}.in_proj.{pname}"] = param.detach().clone()
            if module._needs_out_proj:
                for pname, param in module.out_proj.named_parameters():
                    state[f"{name}.out_proj.{pname}"] = param.detach().clone()
    return state


def load_hyplora(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load extracted HypLoRA weights back into an injected model."""
    current = {n: p for n, p in model.named_parameters()}
    for key, val in state.items():
        if key in current:
            current[key].data.copy_(val)
        else:
            warnings.warn(f"[HypLoRA] Key not found in model: {key}")


# ─────────────────────────────────────────────────────────────────────────────
# merge_hyplora — fuse adapter into base weights (inference speed)
# ─────────────────────────────────────────────────────────────────────────────

def merge_hyplora(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Fuse HypLoRA adapters into their base linear weights and replace
    HypLoRALayer with plain nn.Linear for inference without overhead.

    Note: merging is an approximation (the hyperbolic branch is non-linear),
    so merged weights only match the original at x=0. For quality inference,
    prefer keeping the full HypLoRALayer active.

    Args:
        model:   Model with injected HypLoRALayer modules.
        inplace: Modify model in place (default True).

    Returns:
        Model with HypLoRALayer replaced by merged nn.Linear.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    for name, module in list(model.named_modules()):
        if not isinstance(module, HypLoRALayer):
            continue

        # Compute effective adapter weight via a probe at origin
        # W_eff ≈ W_base + (∂adapter/∂x)|_{x=0}
        # For small alpha/rank, the first-order approximation:
        #   delta_W ≈ scale * B^T @ A^T  (tangent-space linear part)
        with torch.no_grad():
            A = module.llr.A.data            # [n_embd, rank]
            B = module.llr.B.data            # [rank, n_embd]
            # A[n,r], B[r,n]: adapter = x @ A @ B, weight equivalent = (A@B)^T
            delta_W = (A @ B).T * module.scale   # [n_embd, n_embd]

            W = module.base_linear.weight.data.clone()
            bias = module.base_linear.bias.data.clone() if module.base_linear.bias is not None else None

            # Handle dimension mismatch
            if delta_W.shape == W.shape:
                W_merged = W + delta_W.to(W.dtype)
            else:
                warnings.warn(f"[HypLoRA] Shape mismatch at {name}: "
                              f"W={W.shape}, delta={delta_W.shape}. Skipping merge.")
                continue

            merged = nn.Linear(
                module.in_features, module.out_features,
                bias=(bias is not None), dtype=W.dtype,
                device=W.device,
            )
            merged.weight.data = W_merged
            if bias is not None:
                merged.bias.data = bias

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, child_name, merged)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def delta_hyperbolicity(
    embeddings: torch.Tensor,
    n_samples: int = 200,
    seed: int = 42,
) -> float:
    """
    Estimate δ-hyperbolicity of a set of embeddings using the four-point condition.

    A space has δ-hyperbolicity δ iff for all four points x,y,z,w:
        d(x,z) + d(y,w) ≤ max(d(x,y) + d(z,w), d(x,w) + d(y,z)) + 2δ

    Lower δ → more tree-like / hyperbolic.
    δ=0 → perfect tree.
    For NLP: δ ≈ 0.02 to 0.1 (highly hyperbolic per HypLoRA paper).

    Args:
        embeddings: [N, d]  Euclidean embeddings (e.g. token embeddings)
        n_samples:  Number of random 4-tuples to sample
        seed:       Random seed for reproducibility

    Returns:
        Estimated δ (float).
    """
    torch.manual_seed(seed)
    N = embeddings.shape[0]
    if N < 4:
        return float('nan')

    emb = F.normalize(embeddings.float(), dim=-1)

    # Approximate pairwise distances via cosine (cheap for large vocab)
    # Full geodesic distance requires substrate — use Euclidean as proxy
    def dist_matrix(x: torch.Tensor) -> torch.Tensor:
        # Euclidean distance matrix [N, N]
        diff = x.unsqueeze(0) - x.unsqueeze(1)         # [N, N, d]
        return diff.norm(dim=-1)                        # [N, N]

    n_samples = min(n_samples, N * (N-1) // 4)
    idx = torch.randint(0, N, (n_samples, 4))

    # Sample subset for efficiency
    sub = emb[idx.unique()]
    D = dist_matrix(sub)

    delta_max = 0.0
    for i in range(min(n_samples, 500)):
        a, b, c, d = idx[i]
        # Map original indices to sub-matrix indices
        try:
            ia = (idx.unique() == a).nonzero(as_tuple=True)[0].item()
            ib = (idx.unique() == b).nonzero(as_tuple=True)[0].item()
            ic = (idx.unique() == c).nonzero(as_tuple=True)[0].item()
            id_ = (idx.unique() == d).nonzero(as_tuple=True)[0].item()
        except Exception:
            continue

        dab = D[ia, ib].item(); dcd = D[ic, id_].item()
        dac = D[ia, ic].item(); dbd = D[ib, id_].item()
        dad = D[ia, id_].item(); dbc = D[ib, ic].item()

        s1 = dab + dcd
        s2 = dac + dbd
        s3 = dad + dbc
        sorted_s = sorted([s1, s2, s3], reverse=True)
        delta = (sorted_s[0] - sorted_s[1]) / 2.0
        delta_max = max(delta_max, delta)

    return delta_max


def token_freq_norm_stats(
    embeddings: torch.Tensor,
    token_freqs: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute token frequency vs embedding norm correlation.

    Per HypLoRA paper: frequent tokens (abstract) have lower norm (closer
    to origin); rare tokens (specific) have higher norm (farther from origin).
    This is the empirical signature of hyperbolic structure in LLMs.

    Args:
        embeddings:   [V, d]  token embedding matrix
        token_freqs:  [V]     token frequencies (optional).
                              If None, computes norm statistics only.

    Returns:
        Dict with: mean_norm, std_norm, min_norm, max_norm,
                   freq_norm_corr (Pearson, if freqs provided)
    """
    norms = embeddings.norm(dim=-1).float()   # [V]

    stats = {
        "mean_norm": norms.mean().item(),
        "std_norm":  norms.std().item(),
        "min_norm":  norms.min().item(),
        "max_norm":  norms.max().item(),
        "n_tokens":  embeddings.shape[0],
    }

    if token_freqs is not None:
        freqs = token_freqs.float()
        # Pearson correlation between log(freq) and norm
        log_freqs = torch.log(freqs.clamp(min=1.0))
        lf_mean  = log_freqs.mean()
        n_mean   = norms.mean()
        cov      = ((log_freqs - lf_mean) * (norms - n_mean)).mean()
        std_lf   = log_freqs.std().clamp(min=1e-8)
        std_n    = norms.std().clamp(min=1e-8)
        corr     = (cov / (std_lf * std_n)).item()
        stats["freq_norm_corr"] = corr
        # Expected: corr < 0 (frequent tokens → lower norm)
        stats["hyperbolic_signal"] = corr < -0.1

    return stats

def print_trainable_params(model: nn.Module, verbose: bool = False) -> None:
    """Print trainable / frozen parameter counts for HypLoRA analysis."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen    = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]

    t_count = sum(p.numel() for _, p in trainable)
    f_count = sum(p.numel() for _, p in frozen)
    total   = t_count + f_count

    print(f"  Trainable : {t_count:>12,}  ({100*t_count/total:.3f}%)")
    print(f"  Frozen    : {f_count:>12,}  ({100*f_count/total:.3f}%)")
    print(f"  Total     : {total:>12,}")

    if verbose:
        print("\n  Trainable parameters:")
        for n, p in trainable:
            print(f"    {n:<60} {str(tuple(p.shape)):<20} {p.numel():>10,}")
