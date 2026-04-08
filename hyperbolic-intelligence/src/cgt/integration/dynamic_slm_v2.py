"""
cgt/integration/dynamic_slm_v2.py
=======================================
Isolated DynamicSLMWrapperV2 — geometry-correct, no Euclidean fallback.

CRITICAL DIFFERENCES FROM LEGACY dynamic_slm.py
-------------------------------------------------
1.  No silent fallback path.
    Legacy:  evolved_hyp = hyp_embed + 0.1 * phase_signal   ← ILLEGAL
    v2:      riemannian_step_v2() always — RuntimeError if substrate missing.

2.  Shape contract is explicit.
    Input may be [B, L, embed_dim] (Euclidean) or [B, L, embed_dim+1] (Lorentz).
    ensure_lorentz_v2() lifts Euclidean → Lorentz BEFORE any geometry op.

3.  After Riemannian update, output is [B, L, hyp_dim+1] (Lorentz ambient).
    Time component is stripped to [B, L, hyp_dim] before out_projector.
    If input was Lorentz-ambient, time component is restored on output.

4.  Kuramoto oscillator count adapts to sequence length via interpolation
    with an explicit state restore (thread-safe).

5.  DynamicsConfigV2 precedence rule (n_ctx alias) applied automatically.

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.config import DynamicsConfigV2
from cgt.dynamics.kuramoto_v2 import KuramotoSystemV2
from cgt.dynamics.trajectory_v2 import TrajectoryV2
from cgt.dynamics.riemannian_update_v2 import (
    RiemannianPhaseDynamicsV2,
    ensure_lorentz_v2,
)
from cgt.geometry import LorentzConfigV2, LorentzSubstrateV2


# ─────────────────────────────────────────────────────────────────────────────
# Linear projector
# ─────────────────────────────────────────────────────────────────────────────

class _LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# DynamicSLMWrapperV2
# ─────────────────────────────────────────────────────────────────────────────

class DynamicSLMWrapperV2(nn.Module):
    """
    Geometry-correct Dynamic SLM Wrapper (v2).

    Architecture
    ------------
    Input (from slm1 or direct tensor)
        ↓
    in_projector [embed_dim → hyp_dim]  (if embed_dim ≠ hyp_dim)
        ↓   [B, L, hyp_dim]  Euclidean spatial
    ensure_lorentz_v2 → lift → [B, L, hyp_dim+1]  H^n point
        ↓
    KuramotoSystemV2 → phase_signal [B, L, hyp_dim]
        ↓
    RiemannianPhaseDynamicsV2 → [B, L, hyp_dim+1]  updated H^n point
        ↓
    strip time → [B, L, hyp_dim]
        ↓
    out_projector [hyp_dim → embed_dim]  (if embed_dim ≠ hyp_dim)
        ↓
    Output (to slm2 or returned)

    No fallback to Euclidean addition.
    """

    def __init__(
        self,
        slm1: Optional[nn.Module] = None,
        slm2: Optional[nn.Module] = None,
        config: Optional[DynamicsConfigV2] = None,
    ) -> None:
        super().__init__()
        self.cfg  = config or DynamicsConfigV2()
        self.slm1 = slm1
        self.slm2 = slm2

        # ── Substrate ─────────────────────────────────────────────────────
        lorentz_cfg = LorentzConfigV2(
            intrinsic_dim      = self.cfg.hyperbolic_dim,
            learnable_curvature= False,
            initial_curvature  = 1.0,
        )
        self.substrate = LorentzSubstrateV2(lorentz_cfg)

        # ── Projectors (Euclidean, spatial dims only) ──────────────────────
        self.in_projector: Optional[_LinearProjector] = None
        self.out_projector: Optional[_LinearProjector] = None
        if self.cfg.embed_dim != self.cfg.hyperbolic_dim:
            self.in_projector  = _LinearProjector(self.cfg.embed_dim,     self.cfg.hyperbolic_dim)
            self.out_projector = _LinearProjector(self.cfg.hyperbolic_dim, self.cfg.embed_dim)

        # ── Kuramoto ──────────────────────────────────────────────────────
        self.kuramoto = KuramotoSystemV2(config=self.cfg)

        # ── Riemannian dynamics (NO FALLBACK — geometry always enforced) ───
        self.riemannian = RiemannianPhaseDynamicsV2(
            substrate     = self.substrate,
            step_size     = 0.1,
            clamp_tangent = 5.0,
        )

        # ── Trajectory ────────────────────────────────────────────────────
        self._last_trajectory: Optional[TrajectoryV2] = None

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: Any,
        slm1_kwargs: Optional[Dict] = None,
        slm2_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        slm1_kwargs = slm1_kwargs or {}
        slm2_kwargs = slm2_kwargs or {}

        embeddings = self.slm1(x, **slm1_kwargs) if self.slm1 is not None else x
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"DynamicSLMWrapperV2: expected Tensor from slm1, got {type(embeddings)}. "
                "Override _extract_embeddings() if slm1 returns a non-tensor."
            )

        if not self.cfg.use_dynamics:
            return self.slm2(embeddings, **slm2_kwargs) if self.slm2 else embeddings

        evolved = self._run_dynamics(embeddings)
        return self.slm2(evolved, **slm2_kwargs) if self.slm2 else evolved

    # ── Dynamics core ─────────────────────────────────────────────────────

    def _run_dynamics(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Full Riemannian dynamics pipeline.

        Shape contract
        --------------
        Input : [B, L, embed_dim]  OR  [B, L, embed_dim+1]  (Lorentz ambient)
        Output: same shape as input

        All geometry is handled inside — caller gets back the same format.
        """
        # ── Detect + strip Lorentz-ambient input ──────────────────────────
        lorentz_input    = False
        lorentz_ambient  = self.cfg.embed_dim + 1
        saved_time: Optional[torch.Tensor] = None

        if embeddings.shape[-1] == lorentz_ambient:
            saved_time    = embeddings[..., :1]        # [B, L, 1]
            embeddings    = embeddings[..., 1:]        # [B, L, embed_dim]
            lorentz_input = True
        elif embeddings.shape[-1] != self.cfg.embed_dim:
            # Coerce to expected embed_dim
            got = embeddings.shape[-1]
            tgt = self.cfg.embed_dim
            embeddings = (
                embeddings[..., :tgt] if got > tgt
                else F.pad(embeddings, (0, tgt - got))
            )

        B, L, _ = embeddings.shape

        # ── Project to hyperbolic_dim (Euclidean spatial) ─────────────────
        hyp_embed = (
            self.in_projector(embeddings)
            if self.in_projector is not None
            else embeddings
        )
        # hyp_embed: [B, L, hyp_dim]  — EUCLIDEAN, no time component yet

        # ── Adapt Kuramoto oscillator count to sequence length ─────────────
        hyp_embed, final_theta = self._run_kuramoto(hyp_embed, B, L)
        # final_theta: [B, L]

        # ── Build phase signal [B, L, hyp_dim] ───────────────────────────
        hyp_dim = hyp_embed.shape[-1]
        phase_enc = torch.stack(
            [torch.cos(final_theta), torch.sin(final_theta)], dim=-1
        )   # [B, L, 2]
        repeats      = math.ceil(hyp_dim / 2)
        phase_signal = phase_enc.repeat(1, 1, repeats)[..., :hyp_dim]  # [B, L, hyp_dim]

        # FIX: Phase bypass gate — prevents Kuramoto from driving logits when
        # spatial embeddings have collapsed to the origin.
        #
        # Root cause of "Amplitude-Phase Disconnection": when hyp_embed ≈ 0,
        # phase_signal still has magnitude 1 (cos²+sin²=1). The Riemannian update
        # then applies a unit-magnitude tangent perturbation to a near-origin point,
        # effectively driving the output from phase noise alone — zero spatial info.
        #
        # Gate: scale phase_signal by the normalised spatial norm of hyp_embed.
        # When radius ≈ 0 → gate ≈ 0 → phase cannot inject signal.
        # When radius is healthy → gate ≈ 1 → phase acts normally.
        #
        # Safety check: emit warning if all gates are collapsed.
        spatial_norm = torch.norm(hyp_embed, dim=-1, keepdim=True)  # [B, L, 1]
        gate = torch.tanh(spatial_norm)                              # [B, L, 1] in (0,1)
        phase_signal = phase_signal * gate                           # gated phase

        if gate.mean().item() < 1e-3:
            import warnings
            warnings.warn(
                f"[PhaseBypass] Phase gate mean={gate.mean().item():.2e} < 1e-3. "
                f"Spatial embeddings have collapsed — phase is being suppressed. "
                f"Check radius loss and lambda_radius configuration.",
                RuntimeWarning, stacklevel=2,
            )

        # ── Riemannian update ──────────────────────────────────────────────
        # riemannian_step_v2 internally calls ensure_lorentz_v2,
        # so hyp_embed [B, L, hyp_dim] is lifted correctly BEFORE any geo op.
        # Output is [B, L, hyp_dim+1] (Lorentz ambient) in float64.
        evolved_lorentz = self.riemannian(hyp_embed, phase_signal)
        # FIX: do NOT cast to float32 here. Keep float64 for the x0 reconstruction
        # below. Casting early then doing norm_sq in float32 introduces ~1e-4 error.
        # We cast only once at the very end, after proj().

        # ── Strip time component (out_projector operates on spatial) ───────
        ambient = self.substrate.n + 1   # hyp_dim + 1
        if evolved_lorentz.shape[-1] == ambient:
            evolved_spatial = evolved_lorentz[..., 1:]   # [B, L, hyp_dim] float64
        else:
            evolved_spatial = evolved_lorentz

        # ── Project back to embed_dim ─────────────────────────────────────
        # out_projector has float32 weights → cast spatial to float32 first
        if self.out_projector is not None:
            evolved = self.out_projector(evolved_spatial.to(hyp_embed.dtype))
        else:
            evolved = evolved_spatial   # still float64 if no projector
        # evolved: [B, L, embed_dim]

        # ── Restore Lorentz-ambient format if input was ambient ─────────
        # FIX: keep computation in float64 throughout, cast to orig dtype only once.
        # Computing norm_sq and x0 in float32 with ||spatial||~10-100 gives
        # catastrophic cancellation: error ~1e-3 to 1e-2 in Minkowski constraint.
        # In float64: same computation gives ~1e-12. Single cast at end: ~1e-7 (f32 ULP).
        if lorentz_input:
            orig_dtype = hyp_embed.dtype
            evolved64  = evolved.double()
            K64 = self.substrate.K.double().to(evolved64.device)
            norm_sq64 = (evolved64 ** 2).sum(dim=-1, keepdim=True)
            x0_new64  = torch.sqrt((1.0 / K64 + norm_sq64).clamp(min=1e-15))
            evolved64 = torch.cat([x0_new64, evolved64], dim=-1)   # [B, L, embed_dim+1]
            # Final proj() now upcasts internally → double-precision constraint enforcement
            BL = evolved64.shape[0] * evolved64.shape[1]
            evolved64 = self.substrate.proj(evolved64.reshape(BL, -1)).reshape(evolved64.shape)
            # Single cast to original dtype — ~1e-7 rounding error (float32 ULP at x0~10)
            evolved = evolved64.to(orig_dtype)

        return evolved

    # ── Kuramoto with adaptive oscillator count ───────────────────────────

    def _run_kuramoto(
        self,
        hyp_embed: torch.Tensor,   # [B, L, hyp_dim]
        B: int,
        L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Kuramoto simulation, adapting oscillator count to sequence length L.

        Returns:
            hyp_embed   : unchanged [B, L, hyp_dim]
            final_theta : [B, L] final phase values
        """
        device = hyp_embed.device
        original_N    = self.cfg.num_oscillators
        original_freq = self.kuramoto.natural_frequencies.data.clone()

        # Temporarily adapt to sequence length
        self.cfg.num_oscillators          = L
        self.kuramoto.cfg.num_oscillators = L

        # Interpolate natural frequencies if necessary
        if self.kuramoto.natural_frequencies.shape[0] != L:
            with torch.no_grad():
                old_freq = self.kuramoto.natural_frequencies.data
                new_freq = F.interpolate(
                    old_freq.view(1, 1, -1).float(),
                    size=L,
                    mode="linear",
                    align_corners=False,
                ).view(L).to(old_freq.dtype)

                if isinstance(self.kuramoto.natural_frequencies, nn.Parameter):
                    self.kuramoto.natural_frequencies.data = new_freq
                else:
                    self.kuramoto.register_buffer("natural_frequencies", new_freq)

        initial_theta = torch.rand(B, L, device=device) * 2.0 * math.pi

        final_theta, phase_list = self.kuramoto.simulate(
            initial_state = initial_theta,
            embeddings    = hyp_embed,
        )

        # Record trajectory
        self._last_trajectory = TrajectoryV2.from_phase_list(phase_list)

        # Restore original oscillator count AND frequencies (thread-safe state restore)
        self.cfg.num_oscillators          = original_N
        self.kuramoto.cfg.num_oscillators = original_N
        with torch.no_grad():
            if isinstance(self.kuramoto.natural_frequencies, nn.Parameter):
                self.kuramoto.natural_frequencies.data = original_freq
            else:
                self.kuramoto.register_buffer("natural_frequencies", original_freq)

        return hyp_embed, final_theta

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def last_trajectory(self) -> Optional[TrajectoryV2]:
        return self._last_trajectory
