# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Non-Monotonic Coupling Scheduler
=================================

Inspired by: Gu, Lüders & Bechinger (2026). "Non-monotonic magnetic friction
from collective rotor dynamics." Nature Materials.

The existing DynamicCouplingScheduler (coupling.py) uses a monotonic schedule:
    K(t) = K_min + (K_max - K_min) * cosine(t)

This can drive the system through the "friction peak" — the coupling regime
where DegEq is most likely to form. The magnetic friction paper shows that
friction vs coupling strength is NON-MONOTONIC with a clear peak at
intermediate coupling.

This module provides two schedulers:
    1. FrictionAwareCouplingScheduler: monitors frustration and adapts coupling
       to stay BELOW the friction peak.
    2. NonMonotonicCouplingScheduler: prescribes a non-monotonic coupling
       trajectory that sweeps UP to the peak, then retreats.

Design: ADDITIVE ONLY. Use as drop-in replacement for DynamicCouplingScheduler.
No modification to existing modules.

Usage:
    from cgt.dynamics.magnetic_friction_scheduler import (
        FrictionAwareCouplingScheduler,
        NonMonotonicCouplingScheduler,
    )

    scheduler = FrictionAwareCouplingScheduler(
        coupling_min=0.1,
        coupling_max=2.0,
        friction_monitor=frustration_monitor,  # FrustrationOrderParameter
    )

    for step in range(total_steps):
        K = scheduler.get_coupling(phases=current_phases)
        # ... use K in Kuramoto dynamics ...
        scheduler.step()
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FrictionAwareCouplingScheduler(nn.Module):
    """
    Adaptive coupling scheduler that avoids the friction peak.

    Monitors the frustration order parameter F and adjusts coupling
    strength to stay below the DegEq-prone regime.

    Algorithm:
        1. Start with linear warmup to K_max.
        2. Monitor F (frustration) at each step.
        3. If F > F_threshold (approaching friction peak):
           - Reduce coupling by α_retreat per step.
        4. If F < F_low (below friction peak):
           - Allow coupling to increase slowly.
        5. Maintain EMA of frustration for smoothness.

    This is analogous to the "contactless friction control" described in
    Gu et al.: by adjusting the effective "distance" (coupling strength),
    the system avoids the frustrated regime where energy dissipation peaks.

    Args:
        coupling_min: Minimum coupling strength.
        coupling_max: Maximum coupling strength.
        warmup_steps: Steps for initial linear warmup.
        frustration_threshold: F above which coupling is reduced (default 0.5).
        frustration_target: Target F for optimal operation (default 0.25).
        retreat_rate: How fast to reduce coupling when frustrated (default 0.02).
        advance_rate: How fast to increase coupling when not frustrated (default 0.005).
        ema_beta: EMA smoothing for frustration tracking (default 0.95).
    """

    def __init__(
        self,
        coupling_min: float = 0.1,
        coupling_max: float = 2.0,
        warmup_steps: int = 1000,
        frustration_threshold: float = 0.5,
        frustration_target: float = 0.25,
        retreat_rate: float = 0.02,
        advance_rate: float = 0.005,
        ema_beta: float = 0.95,
    ):
        super().__init__()
        self.coupling_min = coupling_min
        self.coupling_max = coupling_max
        self.warmup_steps = warmup_steps
        self.frustration_threshold = frustration_threshold
        self.frustration_target = frustration_target
        self.retreat_rate = retreat_rate
        self.advance_rate = advance_rate
        self.ema_beta = ema_beta

        self.register_buffer("current_step", torch.tensor(0))
        self.register_buffer("current_coupling", torch.tensor(coupling_min))
        self.register_buffer("frustration_ema", torch.tensor(0.0))
        self.register_buffer("regime", torch.tensor(0))
        # regime: 0 = warmup, 1 = advancing, 2 = retreating

        # History for diagnostics
        self.register_buffer(
            "coupling_history",
            torch.zeros(10000),  # ring buffer
        )
        self.register_buffer("history_idx", torch.tensor(0))

    def _update_frustration(self, frustration: float) -> None:
        """Update EMA of frustration."""
        self.frustration_ema = (
            self.ema_beta * self.frustration_ema
            + (1.0 - self.ema_beta) * frustration
        )

    def get_coupling(
        self,
        phases: Optional[torch.Tensor] = None,
        frustration: Optional[float] = None,
    ) -> float:
        """
        Get current coupling strength.

        Args:
            phases: [B, H] phases (optional, for frustration computation).
            frustration: Pre-computed frustration value (overrides phases).

        Returns:
            Current coupling strength K.
        """
        step = self.current_step.item()

        # Warmup phase: linear ramp
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            K = self.coupling_min + (self.coupling_max - self.coupling_min) * alpha
            self.current_coupling.fill_(K)
            self.regime.fill_(0)
            return K

        # Post-warmup: frustration-aware adaptation
        if frustration is not None:
            self._update_frustration(frustration)

        f = self.frustration_ema.item()
        K = self.current_coupling.item()

        if f > self.frustration_threshold:
            # Frustrated regime → retreat (reduce coupling)
            K = max(self.coupling_min, K - self.retreat_rate * (self.coupling_max - self.coupling_min))
            self.regime.fill_(2)
        elif f < self.frustration_target:
            # Below target → advance (increase coupling cautiously)
            K = min(self.coupling_max, K + self.advance_rate * (self.coupling_max - self.coupling_min))
            self.regime.fill_(1)
        # else: in target band → hold steady

        self.current_coupling.fill_(K)
        return K

    def step(self) -> None:
        """Increment step counter and record history."""
        # Record coupling in ring buffer
        idx = self.history_idx.item() % self.coupling_history.shape[0]
        self.coupling_history[idx] = self.current_coupling
        self.history_idx += 1
        self.current_step += 1

    def get_state(self) -> dict:
        """Return current state for logging."""
        regimes = {0: "warmup", 1: "advancing", 2: "retreating"}
        return {
            "step": self.current_step.item(),
            "coupling": self.current_coupling.item(),
            "frustration_ema": self.frustration_ema.item(),
            "regime": regimes.get(self.regime.item(), "unknown"),
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_step.fill_(0)
        self.current_coupling.fill_(self.coupling_min)
        self.frustration_ema.fill_(0.0)
        self.regime.fill_(0)
        self.coupling_history.fill_(0.0)
        self.history_idx.fill_(0)


class NonMonotonicCouplingScheduler(nn.Module):
    """
    Prescribes a non-monotonic coupling trajectory.

    Instead of adaptive frustration monitoring, this scheduler follows a
    FIXED non-monotonic schedule inspired by the friction curve:

        Phase 1 (warmup):    K ramps linearly from K_min to K_max
        Phase 2 (peak):      K held at K_max briefly (exploration)
        Phase 3 (retreat):   K decreases following inverse-sqrt to K_cruise
        Phase 4 (cruise):    K held at K_cruise (stable training regime)

    The rationale: the system needs to BRIEFLY explore the high-coupling
    regime to discover hierarchical structure, then retreat to a lower
    coupling where frustration is manageable.

    This mirrors the experimental protocol in Gu et al. where the
    interlayer distance is systematically varied to map the friction curve.

    Args:
        coupling_min: Minimum coupling (Phase 1 start).
        coupling_max: Maximum coupling (Phase 2 peak).
        coupling_cruise: Stable cruise coupling (Phase 4). Default: 0.4 * K_max.
        warmup_steps: Duration of Phase 1.
        peak_steps: Duration of Phase 2 (exploration at high coupling).
        retreat_steps: Duration of Phase 3 (retreat from peak).
        schedule_type: Retreat shape ('sqrt', 'cosine', 'linear').
    """

    def __init__(
        self,
        coupling_min: float = 0.1,
        coupling_max: float = 2.0,
        coupling_cruise: Optional[float] = None,
        warmup_steps: int = 500,
        peak_steps: int = 200,
        retreat_steps: int = 1000,
        schedule_type: str = 'cosine',
    ):
        super().__init__()
        self.coupling_min = coupling_min
        self.coupling_max = coupling_max
        self.coupling_cruise = coupling_cruise or 0.4 * coupling_max
        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.retreat_steps = retreat_steps
        self.schedule_type = schedule_type

        # Phase boundaries
        self._phase2_start = warmup_steps
        self._phase3_start = warmup_steps + peak_steps
        self._phase4_start = warmup_steps + peak_steps + retreat_steps

        self.register_buffer("current_step", torch.tensor(0))

    def forward(self) -> float:
        """Get current coupling strength."""
        step = self.current_step.item()

        if step < self.warmup_steps:
            # Phase 1: linear warmup
            alpha = step / max(self.warmup_steps, 1)
            return self.coupling_min + (self.coupling_max - self.coupling_min) * alpha

        elif step < self._phase3_start:
            # Phase 2: peak exploration
            return self.coupling_max

        elif step < self._phase4_start:
            # Phase 3: retreat from peak
            t_retreat = step - self._phase3_start
            progress = t_retreat / max(self.retreat_steps, 1)

            if self.schedule_type == 'cosine':
                # Smooth cosine decay from K_max to K_cruise
                alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.coupling_cruise + (self.coupling_max - self.coupling_cruise) * alpha

            elif self.schedule_type == 'sqrt':
                # Inverse sqrt: fast initial retreat, slow tail
                alpha = 1.0 - math.sqrt(progress)
                return self.coupling_cruise + (self.coupling_max - self.coupling_cruise) * alpha

            elif self.schedule_type == 'linear':
                return self.coupling_max - (self.coupling_max - self.coupling_cruise) * progress

            else:
                return self.coupling_cruise

        else:
            # Phase 4: cruise
            return self.coupling_cruise

    def step(self) -> None:
        """Increment step counter."""
        self.current_step += 1

    def get_phase(self) -> str:
        """Return current phase name."""
        step = self.current_step.item()
        if step < self.warmup_steps:
            return "warmup"
        elif step < self._phase3_start:
            return "peak"
        elif step < self._phase4_start:
            return "retreat"
        else:
            return "cruise"

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_step.fill_(0)
