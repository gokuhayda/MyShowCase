# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
HPC Training Guard
==================

Protects GPU-hours from training runs trapped in DegEq.

The HysteresisDetector (cgt.diagnostics.hysteresis) detects DegEq
precursors ~1000 steps before full attractor onset. This guard
converts those detections into actionable HPC decisions:

    OK     → continue normally
    WATCH  → log warning, monitor closely
    SHIELD → force checkpoint, reduce learning rate
    ABORT  → force checkpoint, stop training

Escalation logic:
    1. First hysteresis detection        → WATCH
    2. Sustained hysteresis > patience   → SHIELD
    3. RDC exceeds abort_threshold       → ABORT

Design: PURELY ADDITIVE. No model or optimizer modifications.
Works with any training loop that can provide rdc_ema per step.

Usage:
    from cgt.diagnostics.hpc_guard import HPCTrainingGuard

    guard = HPCTrainingGuard(
        patience=500,
        abort_rdc_threshold=8.0,
    )

    for step in range(total_steps):
        # ... training step ...
        action = guard.step(rdc_ema=rdc_value, step=step)

        if action == "SHIELD":
            save_checkpoint(model, optimizer, step)
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
        elif action == "ABORT":
            save_checkpoint(model, optimizer, step)
            print(guard.report())
            break
"""

from __future__ import annotations

from typing import Dict, List, Optional


class HPCTrainingGuard:
    """
    HPC-aware training protection against DegEq.

    Monitors RDC trajectory via HysteresisDetector and escalates
    through three protection levels when DegEq precursors appear.

    Args:
        patience: Steps of sustained hysteresis before SHIELD.
        abort_rdc_threshold: RDC value that triggers immediate ABORT.
        hysteresis_window: Window size for the internal HysteresisDetector.
        hysteresis_min_amplitude: Min peak-trough amplitude to count as cycle.
    """

    def __init__(
        self,
        patience: int = 500,
        abort_rdc_threshold: float = 8.0,
        hysteresis_window: int = 200,
        hysteresis_min_amplitude: float = 0.5,
    ):
        from cgt.diagnostics.hysteresis import HysteresisDetector

        self.detector = HysteresisDetector(
            window_size=hysteresis_window,
            min_amplitude=hysteresis_min_amplitude,
        )
        self.patience = patience
        self.abort_rdc_threshold = abort_rdc_threshold

        self._hysteresis_onset_step: Optional[int] = None
        self._shield_activated: bool = False
        self._current_step: int = 0
        self._alerts: List[Dict] = []

    def step(
        self,
        rdc_ema: float,
        step: int,
        order_parameter: Optional[float] = None,
        logit_std: Optional[float] = None,
        l_hidden: Optional[float] = None,
    ) -> str:
        """
        Check training health and return action.

        Call once per training step with current metrics.

        Args:
            rdc_ema: Current RDC EMA value (primary signal).
            step: Current training step.
            order_parameter: Kuramoto order parameter (optional).
            logit_std: Logit standard deviation (optional).
            l_hidden: Hidden alignment loss (optional).

        Returns:
            Action: 'OK', 'WATCH', 'SHIELD', or 'ABORT'.
        """
        self._current_step = step

        # Feed detector
        self.detector.update(
            rdc_ema=rdc_ema,
            order_parameter=order_parameter,
            logit_std=logit_std,
            l_hidden=l_hidden,
        )

        # ── ABORT: RDC exceeds hard threshold ──
        if rdc_ema > self.abort_rdc_threshold:
            self._log_alert(step, "ABORT",
                            f"rdc_ema={rdc_ema:.2f} > {self.abort_rdc_threshold}")
            return "ABORT"

        # ── Hysteresis check ──
        if self.detector.hysteresis_detected():
            if self._hysteresis_onset_step is None:
                # First detection
                self._hysteresis_onset_step = step
                self._log_alert(step, "WATCH", "hysteresis first detected")
                return "WATCH"

            duration = step - self._hysteresis_onset_step
            if duration >= self.patience and not self._shield_activated:
                self._shield_activated = True
                self._log_alert(step, "SHIELD",
                                f"hysteresis sustained {duration} steps")
                return "SHIELD"

            return "WATCH"

        # ── Hysteresis cleared ──
        if self._hysteresis_onset_step is not None:
            self._hysteresis_onset_step = None
            self._shield_activated = False

        return "OK"

    def _log_alert(self, step: int, action: str, reason: str) -> None:
        self._alerts.append({
            "step": step,
            "action": action,
            "reason": reason,
        })

    def report(self) -> str:
        """Human-readable guard report."""
        lines = [
            "═══ HPC Training Guard ═══",
            f"  Steps monitored : {self._current_step}",
            f"  Alerts triggered: {len(self._alerts)}",
            f"  Shield activated: {'yes' if self._shield_activated else 'no'}",
        ]
        if self._alerts:
            lines.append("  Recent alerts:")
            for a in self._alerts[-5:]:
                lines.append(f"    step {a['step']:>6}: {a['action']:6s} — {a['reason']}")
        # Include hysteresis summary
        lines.append(f"  Hysteresis cycles: {self.detector.get_cycle_count()}")
        lines.append(f"  Loop area (EMA) : {self.detector.get_loop_area():.4f}")
        lines.append("══════════════════════════")
        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, float]:
        """Metrics dict for logging integration."""
        return {
            "hpc_guard_alerts": len(self._alerts),
            "hpc_guard_shield": float(self._shield_activated),
            "hpc_guard_hyst_cycles": self.detector.get_cycle_count(),
            "hpc_guard_loop_area": self.detector.get_loop_area(),
        }
