"""
cgt/config/dynamics_config_v2.py
=====================================
Isolated DynamicsConfigV2.

Config Precedence Rule (applied to ALL aliased fields in __post_init__)
------------------------------------------------------------------------
    if canonical_value != default:
        use canonical_value
    elif alias_value is not None:
        use alias_value
    else:
        use default

This is deterministic and collision-free.
Zero imports from legacy cgt.*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DynamicsConfigV2:
    """
    Configuration for v2 DynamicSLMWrapper and KuramotoSystemV2.

    Canonical fields (these win over any alias)
    -------------------------------------------
    num_oscillators     : number of Kuramoto oscillators / sequence positions
    embed_dim           : input embedding dimension (Euclidean)
    hyperbolic_dim      : intrinsic hyperbolic dimension (spatial coords only)
    dt                  : ODE integration timestep
    num_steps           : number of integration steps per forward pass
    coupling_strength   : Kuramoto coupling K
    noise_std           : SDE noise level (0 = deterministic ODE)
    use_hyperbolic_coupling : use hyperbolic distances for coupling matrix
    use_dynamics        : enable/disable full dynamics pass
    record_trajectory   : store full phase trajectory for analysis
    learnable_frequencies : make natural frequencies nn.Parameters

    Legacy alias fields (accepted, overridden by canonical if set)
    --------------------------------------------------------------
    n_ctx               : alias for num_oscillators
    hidden_dim          : alias for embed_dim
    hyp_dim             : alias for hyperbolic_dim
    """

    # ── canonical ──────────────────────────────────────────────────────────
    num_oscillators: int        = 32    # default 32; alias n_ctx overrides if canonical is default
    embed_dim: int              = 128
    hyperbolic_dim: int         = 128
    dt: float                   = 0.02
    num_steps: int              = 5
    coupling_strength: float    = 0.1
    noise_std: float            = 0.0
    use_hyperbolic_coupling: bool = False
    use_dynamics: bool          = True
    record_trajectory: bool     = False
    learnable_frequencies: bool = True

    # ── T-DPO (disabled by default) ────────────────────────────────────────
    enable_tdpo: bool           = False
    tdpo_beta: float            = 0.1
    tdpo_alpha: float           = 0.5

    # ── legacy aliases ─────────────────────────────────────────────────────
    n_ctx: Optional[int]        = None    # alias → num_oscillators
    hidden_dim: Optional[int]   = None    # alias → embed_dim
    hyp_dim: Optional[int]      = None    # alias → hyperbolic_dim

    # ── precedence sentinels (defaults for comparison) ─────────────────────
    _DEFAULT_NUM_OSC: int       = field(default=32, init=False, repr=False)
    _DEFAULT_EMBED:   int       = field(default=128, init=False, repr=False)
    _DEFAULT_HYP:     int       = field(default=128, init=False, repr=False)

    def __post_init__(self) -> None:
        """Apply precedence rule to all aliased fields."""

        # num_oscillators  ←  canonical > n_ctx > default
        if self.num_oscillators != self._DEFAULT_NUM_OSC:
            pass  # canonical wins
        elif self.n_ctx is not None:
            self.num_oscillators = self.n_ctx

        # embed_dim  ←  canonical > hidden_dim > default
        if self.embed_dim != self._DEFAULT_EMBED:
            pass
        elif self.hidden_dim is not None:
            self.embed_dim = self.hidden_dim

        # hyperbolic_dim  ←  canonical > hyp_dim > default
        if self.hyperbolic_dim != self._DEFAULT_HYP:
            pass
        elif self.hyp_dim is not None:
            self.hyperbolic_dim = self.hyp_dim
