# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Geometric Controller — Closed-Loop Feedback for Hyperbolic Training
====================================================================

The missing piece that connects SENSORS to ACTUATORS.

Sensors (already implemented):
    - HysteresisDetector   → hysteresis_active, loop_area, cycle_count
    - rdc_ema              → radial drift coefficient (from training loop)
    - FrustrationOrderParameter → frustration F, sublattice coherence
    - PhaseTrajectoryAnalyzer   → rotation pattern, velocity correlation

Actuators (already implemented):
    - loss_fn.lambda_hidden, lambda_radius, lambda_contrast
    - VolumeWeightedCE/KL strength
    - FrustrationAwareCoupling.frustration_strength
    - optimizer learning rate
    - coupling strength K

This controller: φ(telemetry) → actuator adjustments

    λ_i(t+1) = λ_i(t) * φ_i(rdc_ema, frustration, hysteresis, pattern)

Design principles:
    1. SOFT adjustments — multiplicative, bounded, EMA-smoothed.
    2. COMPOSABLE with LossBalancer — operates on different signals.
       LossBalancer equalizes gradient magnitudes (proactive balance).
       GeometricController reacts to GEOMETRIC health (DegEq defense).
    3. CONSERVATIVE defaults — strength=0 means no intervention.
    4. ADDITIVE — does not modify any existing module.

Control law summary:
    High rdc_ema    → boost λ_radius, boost volume_weight_strength
    High frustration → reduce coupling_K (retreat from friction peak)
    Hysteresis on   → boost λ_contrast (diversify), reduce lr
    Pattern=staggered → reduce coupling (frustrated regime)
    Pattern=synchronized → increase coupling (under-coupled)

Usage:
    from cgt.dynamics.geometric_controller import GeometricController

    controller = GeometricController(loss_fn=loss_fn)

    for step in range(total_steps):
        # ... forward, compute losses ...
        telemetry = {
            'rdc_ema': rdc_value,
            'frustration': frust_value,
            'hysteresis_active': detector.hysteresis_detected(),
            'order_parameter': r_value,
        }
        adjustments = controller.step(telemetry, step)
        # adjustments are ALREADY applied to loss_fn attributes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math


@dataclass
class ControllerConfig:
    """Configuration for the geometric controller."""

    # ── Activation ──
    warmup_steps: int = 500           # No intervention during warmup
    update_every: int = 10            # Steps between control updates

    # ── RDC response ──
    rdc_safe: float = 2.0             # Below this: no RDC intervention
    rdc_warning: float = 5.0          # Above this: moderate intervention
    rdc_critical: float = 8.0         # Above this: aggressive intervention
    rdc_radius_boost: float = 0.1     # Max multiplicative boost to λ_radius per update
    rdc_volume_boost: float = 0.05    # Max boost to volume_weight strength per update

    # ── Frustration response ──
    frustration_target: float = 0.25  # Ideal frustration level
    frustration_ceiling: float = 0.6  # Above this: retreat coupling
    coupling_retreat_rate: float = 0.05  # Max coupling reduction per update

    # ── Hysteresis response ──
    hysteresis_contrast_boost: float = 0.08  # Boost to λ_contrast when hysteresis
    hysteresis_lr_decay: float = 0.98        # LR multiplier when hysteresis sustained

    # ── Bounds (same format as LossBalancer) ──
    lambda_hidden_bounds: tuple = (0.05, 0.40)
    lambda_radius_bounds: tuple = (0.02, 0.30)
    lambda_contrast_bounds: tuple = (0.03, 0.25)
    volume_strength_bounds: tuple = (0.0, 1.0)
    coupling_bounds: tuple = (0.1, 3.0)

    # ── Smoothing ──
    ema_beta: float = 0.9            # EMA for telemetry smoothing


class GeometricController:
    """
    Closed-loop feedback controller for hyperbolic training dynamics.

    Reads telemetry from diagnostic sensors and adjusts loss weights,
    coupling parameters, and learning rate to prevent DegEq.

    The controller is COMPOSABLE with LossBalancer:
    - LossBalancer: equalizes gradient magnitudes (runs always)
    - GeometricController: reacts to geometric health (DegEq defense)

    Both write to the same loss_fn attributes. GeometricController
    respects LossBalancer bounds and applies soft multiplicative
    adjustments that compose naturally.

    Args:
        loss_fn: The loss module whose λ attributes will be adjusted.
            Expected attributes: lambda_hidden, lambda_radius,
            lambda_contrast (standard HyDRA/CGT distillation loss).
        optimizer: Optional optimizer for LR adjustment.
        config: ControllerConfig with thresholds and rates.
    """

    def __init__(
        self,
        loss_fn: Any = None,
        optimizer: Any = None,
        config: Optional[ControllerConfig] = None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cfg = config or ControllerConfig()

        # ── Smoothed telemetry ──
        self._ema: Dict[str, float] = {
            'rdc': 0.0,
            'frustration': 0.0,
            'order_parameter': 0.5,
            'hysteresis_duration': 0,
        }

        # ── State ──
        self._step: int = 0
        self._hysteresis_onset: Optional[int] = None
        self._log: List[Dict] = []

    def _smooth(self, key: str, value: float) -> float:
        """Update and return EMA-smoothed telemetry value."""
        beta = self.cfg.ema_beta
        if key not in self._ema:
            self._ema[key] = value
        else:
            self._ema[key] = beta * self._ema[key] + (1.0 - beta) * value
        return self._ema[key]

    def _clamp(self, value: float, bounds: tuple) -> float:
        """Clamp to bounds."""
        return max(bounds[0], min(bounds[1], value))

    def _get_attr(self, name: str, default: float) -> float:
        """Safely read attribute from loss_fn."""
        if self.loss_fn is None:
            return default
        val = getattr(self.loss_fn, name, default)
        if hasattr(val, 'item'):
            return val.item()
        return float(val)

    def _set_attr(self, name: str, value: float) -> None:
        """Safely write attribute to loss_fn."""
        if self.loss_fn is not None and hasattr(self.loss_fn, name):
            setattr(self.loss_fn, name, value)

    def step(
        self,
        telemetry: Dict[str, float],
        step: int,
    ) -> Dict[str, float]:
        """
        Execute one control cycle.

        Reads telemetry, computes adjustments, applies them to loss_fn
        and optimizer, returns a dict of what was changed.

        Args:
            telemetry: Dict with any subset of:
                rdc_ema (float): current radial drift coefficient
                frustration (float): frustration order parameter F
                hysteresis_active (bool/float): whether hysteresis detected
                order_parameter (float): Kuramoto order parameter r
                pattern (str): rotation pattern from PhaseTrajectoryAnalyzer
                logit_std (float): logit standard deviation
                l_hidden (float): hidden alignment loss value
            step: Current training step.

        Returns:
            Dict of applied adjustments (empty if no changes).
        """
        self._step = step
        adjustments: Dict[str, float] = {}

        # Skip during warmup
        if step < self.cfg.warmup_steps:
            return adjustments

        # Skip if not on schedule
        if step % self.cfg.update_every != 0:
            return adjustments

        # ── 1. Smooth telemetry ──
        rdc = self._smooth('rdc', telemetry.get('rdc_ema', 0.0))
        frust = self._smooth('frustration', telemetry.get('frustration', 0.0))
        r_order = self._smooth('order_parameter', telemetry.get('order_parameter', 0.5))
        hysteresis = bool(telemetry.get('hysteresis_active', False))
        pattern = telemetry.get('pattern', 'unknown')

        # ── 2. RDC response ──
        if rdc > self.cfg.rdc_safe:
            rdc_severity = min(1.0, (rdc - self.cfg.rdc_safe) /
                               (self.cfg.rdc_critical - self.cfg.rdc_safe))

            # Boost λ_radius: push radial anchor harder
            lam_r = self._get_attr('lambda_radius', 0.05)
            boost = 1.0 + self.cfg.rdc_radius_boost * rdc_severity

            # ── Coherence-Curvature coupling (GCM Theorem A.5) ──────
            # κ ∝ Γ²/(1−Γ²)  →  when Γ↑, geometry regularisation↑
            # This formalises the heuristic rdc-based boost.
            gamma = float(self._ema.get('order_parameter', 0.5))
            gamma_sq = gamma * gamma
            coherence_scale = 1.0 + gamma_sq / max(1.0 - gamma_sq, 0.05)
            boost = boost * coherence_scale  # amplify when coherent

            new_lam_r = self._clamp(lam_r * boost, self.cfg.lambda_radius_bounds)
            self._set_attr('lambda_radius', new_lam_r)
            adjustments['lambda_radius'] = new_lam_r
            adjustments['coherence_scale'] = round(coherence_scale, 3)

            # Boost volume weight strength (if loss supports it)
            vol_str = self._get_attr('volume_weight_strength', 0.5)
            vol_boost = self.cfg.rdc_volume_boost * rdc_severity
            new_vol = self._clamp(vol_str + vol_boost, self.cfg.volume_strength_bounds)
            self._set_attr('volume_weight_strength', new_vol)
            adjustments['volume_weight_strength'] = new_vol

        # ── 3. Frustration response ──
        if frust > self.cfg.frustration_ceiling:
            # Too frustrated: retreat coupling
            overshoot = (frust - self.cfg.frustration_ceiling) / \
                        (1.0 - self.cfg.frustration_ceiling + 1e-6)
            retreat = 1.0 - self.cfg.coupling_retreat_rate * overshoot

            coupling = self._get_attr('coupling_strength', 1.0)
            new_coupling = self._clamp(coupling * retreat, self.cfg.coupling_bounds)
            self._set_attr('coupling_strength', new_coupling)
            adjustments['coupling_strength'] = new_coupling

        elif frust < self.cfg.frustration_target * 0.5:
            # Under-frustrated: gently increase coupling
            advance = 1.0 + self.cfg.coupling_retreat_rate * 0.3
            coupling = self._get_attr('coupling_strength', 1.0)
            new_coupling = self._clamp(coupling * advance, self.cfg.coupling_bounds)
            self._set_attr('coupling_strength', new_coupling)
            adjustments['coupling_strength'] = new_coupling

        # ── 4. Hysteresis response ──
        if hysteresis:
            if self._hysteresis_onset is None:
                self._hysteresis_onset = step

            duration = step - self._hysteresis_onset

            # Boost λ_contrast to diversify representations
            lam_c = self._get_attr('lambda_contrast', 0.05)
            boost = 1.0 + self.cfg.hysteresis_contrast_boost
            new_lam_c = self._clamp(lam_c * boost, self.cfg.lambda_contrast_bounds)
            self._set_attr('lambda_contrast', new_lam_c)
            adjustments['lambda_contrast'] = new_lam_c

            # Decay LR if sustained
            if duration > 200 and self.optimizer is not None:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= self.cfg.hysteresis_lr_decay
                adjustments['lr_decay'] = self.cfg.hysteresis_lr_decay
        else:
            self._hysteresis_onset = None

        # ── 5. Pattern response ──
        if pattern == 'staggered':
            # Frustrated phase dynamics: reduce coupling
            coupling = self._get_attr('coupling_strength', 1.0)
            new_coupling = self._clamp(coupling * 0.97, self.cfg.coupling_bounds)
            self._set_attr('coupling_strength', new_coupling)
            adjustments['coupling_from_pattern'] = new_coupling

        elif pattern == 'synchronized' and r_order > 0.9:
            # Over-synchronized: heads collapsed, boost contrast
            lam_c = self._get_attr('lambda_contrast', 0.05)
            new_lam_c = self._clamp(lam_c * 1.05, self.cfg.lambda_contrast_bounds)
            self._set_attr('lambda_contrast', new_lam_c)
            adjustments['lambda_contrast_sync'] = new_lam_c

        # ── 6. Log ──
        if adjustments:
            record = {
                'step': step,
                'rdc': round(rdc, 3),
                'frustration': round(frust, 3),
                'hysteresis': hysteresis,
                'pattern': pattern,
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in adjustments.items()},
            }
            self._log.append(record)

        return adjustments

    def get_state(self) -> Dict[str, float]:
        """Current controller state for logging."""
        return {
            'ctrl_rdc_ema': self._ema.get('rdc', 0.0),
            'ctrl_frustration_ema': self._ema.get('frustration', 0.0),
            'ctrl_order_ema': self._ema.get('order_parameter', 0.5),
            'ctrl_hysteresis_onset': self._hysteresis_onset or -1,
            'ctrl_interventions': len(self._log),
        }

    def report(self) -> str:
        """Summary of controller activity."""
        lines = [
            "═══ Geometric Controller ═══",
            f"  Steps: {self._step}",
            f"  Interventions: {len(self._log)}",
            f"  Smoothed RDC: {self._ema.get('rdc', 0):.3f}",
            f"  Smoothed frustration: {self._ema.get('frustration', 0):.3f}",
        ]
        if self._log:
            lines.append("  Last 5 interventions:")
            for entry in self._log[-5:]:
                step = entry.pop('step', '?')
                desc = ', '.join(f"{k}={v}" for k, v in entry.items()
                                if k not in ('rdc', 'frustration', 'hysteresis', 'pattern'))
                lines.append(f"    step {step}: {desc}")
                entry['step'] = step  # restore
        lines.append("════════════════════════════")
        return "\n".join(lines)
