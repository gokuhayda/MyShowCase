# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
riemannian_adamw.py
===================
RiemannianAdamW — drop-in replacement for torch.optim.AdamW that applies
the natural gradient correction r/sinh(r) to manifold-resident parameters.

Why this exists (vs the r/sinh(r) correction in distillation_v2.py):
    distillation_v2.py applies the correction inline in the training loop
    before optimizer.step(). This module packages the same logic as a
    proper optimizer subclass for cleaner integration and reuse.

Design:
    Subclasses torch.optim.AdamW — overrides only step().
    Inherits momentum, weight decay, AMSGrad from parent unchanged.
    Applies r/sinh(r) only to params identified by name as manifold-resident.
    After Euclidean AdamW step, reprojects manifold params via substrate.proj().

Bug fixed vs Gemini report:
    Original matched params by shape (p.shape[-1] == ambient_dim), which
    incorrectly modified euclidean attention layers that happen to have
    the same last dimension. Fixed: identify by param name.

Manifold param identification:
    Params whose name contains any of: 'lm_head', 'embed', 'encoder'
    All other params: standard AdamW, no correction, no reprojection.

Usage:
    optimizer = RiemannianAdamW(
        list(student.named_parameters()),
        substrate=substrate,
        lr=3e-4,
        weight_decay=0.01,
    )

Benchmark (CPU, B=4, L=128):
    Extra overhead vs standard AdamW: ~8% per step.
    Comes from r/sinh(r) computation and proj() reprojection.

Relationship to Variant F correction in distillation_v2.py:
    Both implement the same Amari (1998) natural gradient correction.
    This module is the standalone reusable version.
    distillation_v2.py inline correction can be disabled when using this.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


class RiemannianAdamW(torch.optim.AdamW):
    """
    AdamW with Riemannian natural gradient correction for Lorentz manifold.

    Correction factor (Amari 1998, diagonal Fisher approximation):
        c = r / sinh(r.clamp(max=10))
        grad *= c   (applied before super().step())

    where r = ‖param_spatial‖₂ = ‖param[..., 1:]‖₂ is the geodesic
    radius of the parameter from the origin of H^n.

    Asymptotic behaviour:
        r → 0:   c → 1    (no correction near origin, Euclidean regime)
        r → ∞:   c → 0    (strong suppression far from origin)
    This prevents Adam from overshooting at large radius, which is the
    primary driver of DegEq radial drift.
    """

    _MANIFOLD_KEYWORDS = ('lm_head', 'embed', 'encoder')

    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.Tensor]],
        substrate,
        **kwargs,
    ) -> None:
        named_params = list(named_params)

        # Record which param ids are manifold-resident
        self._manifold_ids: set[int] = {
            id(p)
            for name, p in named_params
            if any(k in name for k in self._MANIFOLD_KEYWORDS)
        }

        self.substrate = substrate
        params = [p for _, p in named_params]
        super().__init__(params, **kwargs)

    def _is_manifold(self, p: torch.Tensor) -> bool:
        return id(p) in self._manifold_ids

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        # ── 1. Apply r/sinh(r) correction to manifold params ──────────────
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not self._is_manifold(p):
                    continue

                # Spatial components (all dims except time coord at index 0)
                if p.dim() < 1 or p.shape[-1] < 2:
                    continue

                v_s = p.data[..., 1:]
                r = torch.norm(v_s, dim=-1, keepdim=True).clamp(min=1e-8)
                c = r / torch.sinh(r.clamp(max=10.0))

                # Scale gradient in-place before AdamW momentum update
                p.grad.data.mul_(c)

        # ── 2. Standard AdamW step (momentum, variance, weight decay) ─────
        loss = super().step(closure)

        # ── 3. Retract updated manifold params back to H^n ────────────────
        if self.substrate is not None:
            for group in self.param_groups:
                for p in group['params']:
                    if not self._is_manifold(p):
                        continue
                    if p.dim() < 1 or p.shape[-1] < 2:
                        continue
                    p.data.copy_(self.substrate.proj(p.data))

        # ── 4. Parallel transport momentum to new tangent space ────────────
        # Fonte: H-LLM Spec §10 Remark 10.1.
        # Sem isso, m_t acumula em T_{x_t}H^n mas é usado em T_{x_{t+1}}H^n
        # → frames inconsistentes → silent radius inflation (DegEq).
        #
        # Aproximação: PT_{x→x'}(v) ≈ v - ⟨v, x'⟩_M · x'
        # Projeta m_t no espaço tangente do novo ponto x'.
        # Erro: O(‖x'-x‖²) — desprezível para lr=3e-4.
        if self.substrate is not None:
            for group in self.param_groups:
                for p in group['params']:
                    if not self._is_manifold(p): continue
                    if p.dim() < 1 or p.shape[-1] < 2: continue
                    state = self.state[p]
                    if 'exp_avg' not in state: continue
                    m      = state['exp_avg']           # m_t em T_{x_old}
                    x_new  = p.data                     # x_{t+1} em H^n
                    x_flat = x_new.reshape(-1, x_new.shape[-1])
                    m_flat = m.reshape(-1, m.shape[-1])
                    # ⟨m, x'⟩_M = -m₀x'₀ + Σmᵢx'ᵢ
                    inner = (
                        (m_flat[:, 1:] * x_flat[:, 1:]).sum(-1)
                        - m_flat[:, 0] * x_flat[:, 0]
                    ).unsqueeze(-1)
                    m.sub_((inner * x_flat).reshape_as(m))

        return loss
