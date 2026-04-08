"""
cgt/dynamics/kuramoto_v2.py
================================
Isolated Kuramoto oscillator system for v2.

dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)

Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.config import DynamicsConfigV2


class KuramotoSystemV2(nn.Module):
    """
    Kuramoto oscillator system with discrete-time forward Euler integration.

    State contract
    --------------
    natural_frequencies : [N]   — one per oscillator
    phase_state         : [1, N] — persistent across calls (detached CPU)

    Forward
    -------
    simulate(initial_state, embeddings) → (final_theta, phase_list)
        initial_state : [B, N] initial phases
        embeddings    : [B, N, D] — used to build coupling matrix when
                        use_hyperbolic_coupling=False (cosine similarity)
        Returns:
            final_theta : [B, N] final phases
            phase_list  : List[[B, N]]  — every step's phase values
    """

    def __init__(self, config: Optional[DynamicsConfigV2] = None) -> None:
        super().__init__()
        self.cfg = config or DynamicsConfigV2()
        N = self.cfg.num_oscillators

        if self.cfg.learnable_frequencies:
            self.natural_frequencies = nn.Parameter(torch.randn(N) * 0.1)
        else:
            self.register_buffer("natural_frequencies", torch.randn(N) * 0.1)

        self.register_buffer("phase_state", torch.rand(1, N) * 2.0 * math.pi)

    # ── coupling ──────────────────────────────────────────────────────────

    def _build_coupling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Build [B, N, N] coupling matrix from cosine similarity of embeddings.
        All values in [0, 1] — no negative coupling.
        """
        # embeddings: [B, N, D]
        emb_norm = F.normalize(embeddings, p=2, dim=-1)          # [B, N, D]
        coupling = torch.bmm(emb_norm, emb_norm.transpose(1, 2)) # [B, N, N]
        return coupling.clamp(min=0.0)

    # ── simulate ──────────────────────────────────────────────────────────

    def simulate(
        self,
        initial_state: torch.Tensor,           # [B, N]
        embeddings: Optional[torch.Tensor] = None,  # [B, N, D]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Simulate Kuramoto dynamics for cfg.num_steps steps.

        Returns final phases and (if record_trajectory) all intermediate phases.
        """
        device = initial_state.device
        B, N = initial_state.shape

        theta = initial_state.clone()
        omega = self.natural_frequencies.to(device)

        if embeddings is not None:
            coupling = self._build_coupling(embeddings)   # [B, N, N]
        else:
            coupling = torch.ones(B, N, N, device=device) / N

        phase_list: List[torch.Tensor] = []

        for _ in range(self.cfg.num_steps):
            # Phase difference matrix [B, N, N]
            phase_diff = theta.unsqueeze(2) - theta.unsqueeze(1)

            # Coupling sum [B, N]
            coupling_sum = (coupling * torch.sin(phase_diff)).sum(dim=2)

            # Kuramoto ODE step
            dtheta = omega + (self.cfg.coupling_strength / N) * coupling_sum

            # Optional SDE noise
            if self.cfg.noise_std > 0:
                noise = self.cfg.noise_std * torch.randn_like(theta)
                dtheta = dtheta + noise

            theta = theta + self.cfg.dt * dtheta
            theta = torch.fmod(theta, 2.0 * math.pi)

            if self.cfg.record_trajectory:
                phase_list.append(theta.detach().cpu())

        # Persist state
        with torch.no_grad():
            self.phase_state = theta[:1].detach().cpu()

        return theta, phase_list

    def reset_phases(self) -> None:
        N = self.cfg.num_oscillators
        self.phase_state = torch.rand(1, N) * 2.0 * math.pi
