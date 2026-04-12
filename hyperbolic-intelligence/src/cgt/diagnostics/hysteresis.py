# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Hysteresis Detection for DegEq Early Warning
=============================================

Inspired by: Gu, Lüders & Bechinger (2026). "Non-monotonic magnetic friction
from collective rotor dynamics." Nature Materials.

Key insight from Gu et al.:
    Energy dissipation in the magnetic rotor system arises from HYSTERETIC
    torque cycles: as the layers slide, the rotors repeatedly switch between
    metastable configurations. The area enclosed by the hysteresis loop
    quantifies the energy dissipated per cycle.

    In H-AKORN training dynamics, an analogous hysteresis can appear:
    - The Kuramoto phases oscillate between competing configurations
    - The RDC (radial drift coefficient) oscillates without convergence
    - The order parameter switches between high/low values

    Detecting these hysteretic cycles BEFORE DegEq fully forms provides
    an early warning system — analogous to detecting the onset of the
    frustrated regime before friction peaks.

This module provides:
    1. HysteresisDetector: monitors phase/RDC trajectories for cyclic patterns
    2. MagneticFrictionDiagnostic: comprehensive DegEq risk assessment using
       friction-inspired metrics
    3. Integration with existing DegEqDiagnostics (additive extension)

Design: PURELY ADDITIVE — no modification to existing training or diagnostic code.

Usage:
    from cgt.diagnostics.hysteresis import HysteresisDetector, MagneticFrictionDiagnostic

    # During training loop:
    detector = HysteresisDetector(window_size=200)

    for step in range(total_steps):
        # ... training step ...
        detector.update(
            rdc_ema=rdc_ema_value,
            order_parameter=r_value,
            logit_std=logit_std_value,
            l_hidden=l_hidden_value,
        )

        if detector.hysteresis_detected():
            print(f"⚠ Hysteresis detected at step {step} — DegEq risk HIGH")
            print(detector.report())

    # Post-training:
    diag = MagneticFrictionDiagnostic(detector)
    full_report = diag.run()

References:
    Gu, Lüders & Bechinger (2026). Nature Materials. DOI:10.1038/s41563-026-02538-1
    de Sena (2026). HyDRA v4: Degenerate Equilibrium in Hyperbolic Distillation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Hysteresis Detector
# ─────────────────────────────────────────────────────────────────────────────

class HysteresisDetector:
    """
    Detects hysteretic cycling in training dynamics.

    Monitors a scalar trajectory (e.g. rdc_ema, order parameter) and detects
    when it exhibits oscillatory behavior characteristic of hysteresis:

    1. PEAK/TROUGH detection: identifies local extrema in the trajectory.
    2. CYCLE detection: identifies sequences of peak→trough→peak (or reverse).
    3. LOOP AREA estimation: approximates the area enclosed by the hysteresis
       loop, analogous to energy dissipation in Gu et al.
    4. STATIONARITY test: checks whether the cycle amplitude is stable
       (true hysteresis) vs decaying (transient oscillation).

    The key distinction:
    - Transient oscillation: amplitude decays → system is converging.
    - True hysteresis: amplitude is stable or growing → system is TRAPPED
      in a frustrated equilibrium (DegEq precursor).

    From Gu et al., the magnetic rotors exhibit hysteresis ONLY in the
    intermediate-distance regime where frustration peaks. Similarly,
    training hysteresis should appear only when the system is in the
    DegEq-prone coupling regime.

    Args:
        window_size: Number of steps to monitor for cycle detection.
        min_amplitude: Minimum peak-trough amplitude to count as a cycle.
        stationarity_threshold: Max decay rate to classify as stationary (hysteretic).
        smoothing_beta: EMA smoothing for noisy signals.
    """

    def __init__(
        self,
        window_size: int = 200,
        min_amplitude: float = 0.5,
        stationarity_threshold: float = 0.1,
        smoothing_beta: float = 0.9,
    ):
        self.window_size = window_size
        self.min_amplitude = min_amplitude
        self.stationarity_threshold = stationarity_threshold
        self.smoothing_beta = smoothing_beta

        # Trajectory buffers
        self._rdc_history: deque = deque(maxlen=window_size)
        self._order_history: deque = deque(maxlen=window_size)
        self._logit_std_history: deque = deque(maxlen=window_size)
        self._l_hidden_history: deque = deque(maxlen=window_size)
        self._step_count: int = 0

        # Detected extrema
        self._rdc_peaks: List[Tuple[int, float]] = []
        self._rdc_troughs: List[Tuple[int, float]] = []

        # Cycle analysis cache
        self._cycles: List[Dict] = []
        self._hysteresis_active: bool = False
        self._loop_area_ema: float = 0.0

    def update(
        self,
        rdc_ema: float,
        order_parameter: Optional[float] = None,
        logit_std: Optional[float] = None,
        l_hidden: Optional[float] = None,
    ) -> None:
        """
        Record one training step's metrics.

        Args:
            rdc_ema: Current RDC EMA value (primary signal).
            order_parameter: Kuramoto order parameter r (optional).
            logit_std: Logit standard deviation (optional).
            l_hidden: Hidden alignment loss (optional).
        """
        self._rdc_history.append(rdc_ema)
        if order_parameter is not None:
            self._order_history.append(order_parameter)
        if logit_std is not None:
            self._logit_std_history.append(logit_std)
        if l_hidden is not None:
            self._l_hidden_history.append(l_hidden)

        self._step_count += 1

        # Detect extrema every 10 steps (avoid noise)
        if self._step_count % 10 == 0 and len(self._rdc_history) >= 30:
            self._detect_extrema()
            self._analyze_cycles()

    def _detect_extrema(self) -> None:
        """
        Detect peaks and troughs in the RDC trajectory.

        Uses a simple three-point comparison with EMA smoothing to
        reduce false positives from gradient noise.
        """
        history = list(self._rdc_history)
        n = len(history)
        if n < 5:
            return

        # Smooth the trajectory
        smoothed = [history[0]]
        beta = self.smoothing_beta
        for i in range(1, n):
            smoothed.append(beta * smoothed[-1] + (1.0 - beta) * history[i])

        # Find new extrema (only check recent additions)
        check_start = max(2, n - 15)
        for i in range(check_start, n - 2):
            step_global = self._step_count - (n - i)

            # Peak: smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                # Check minimum amplitude from nearest trough
                if self._rdc_troughs:
                    amp = smoothed[i] - self._rdc_troughs[-1][1]
                    if amp >= self.min_amplitude:
                        # Avoid duplicate detection
                        if not self._rdc_peaks or self._rdc_peaks[-1][0] < step_global - 5:
                            self._rdc_peaks.append((step_global, smoothed[i]))

            # Trough: smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                if self._rdc_peaks:
                    amp = self._rdc_peaks[-1][1] - smoothed[i]
                    if amp >= self.min_amplitude:
                        if not self._rdc_troughs or self._rdc_troughs[-1][0] < step_global - 5:
                            self._rdc_troughs.append((step_global, smoothed[i]))

    def _analyze_cycles(self) -> None:
        """
        Identify complete hysteresis cycles and compute loop area.

        A cycle is: peak → trough → peak (or trough → peak → trough).
        The "loop area" is estimated as:
            A ≈ |amplitude_1 - amplitude_2| × period

        If A is stable across cycles → true hysteresis (DegEq precursor).
        If A decays → transient oscillation (healthy convergence).
        """
        # Need at least 2 peaks and 1 trough (or vice versa) for a cycle
        if len(self._rdc_peaks) < 2 or len(self._rdc_troughs) < 1:
            return

        # Build cycles from recent extrema
        # Merge peaks and troughs into time-ordered sequence
        all_extrema = (
            [(s, v, 'peak') for s, v in self._rdc_peaks[-10:]]
            + [(s, v, 'trough') for s, v in self._rdc_troughs[-10:]]
        )
        all_extrema.sort(key=lambda x: x[0])

        # Extract cycles: any three consecutive extrema form a half-cycle
        new_cycles = []
        for i in range(len(all_extrema) - 2):
            s0, v0, t0 = all_extrema[i]
            s1, v1, t1 = all_extrema[i + 1]
            s2, v2, t2 = all_extrema[i + 2]

            # Must alternate: peak-trough-peak or trough-peak-trough
            if t0 == t1 or t1 == t2:
                continue

            amplitude = abs(v0 - v1)
            period = s2 - s0
            loop_area = amplitude * period

            new_cycles.append({
                "start_step": s0,
                "end_step": s2,
                "amplitude": amplitude,
                "period": period,
                "loop_area": loop_area,
                "type": f"{t0}-{t1}-{t2}",
            })

        if new_cycles:
            self._cycles = new_cycles

        # Determine if hysteresis is active
        if len(self._cycles) >= 2:
            # Check stationarity: are cycle amplitudes stable?
            amplitudes = [c["amplitude"] for c in self._cycles[-5:]]
            if len(amplitudes) >= 2:
                # Compute decay rate: (last - first) / first
                decay = (amplitudes[-1] - amplitudes[0]) / max(abs(amplitudes[0]), 1e-6)
                self._hysteresis_active = abs(decay) < self.stationarity_threshold

                # Update loop area EMA
                latest_area = self._cycles[-1]["loop_area"]
                self._loop_area_ema = (
                    0.8 * self._loop_area_ema + 0.2 * latest_area
                )

    def hysteresis_detected(self) -> bool:
        """
        Returns True if the training dynamics show hysteretic cycling.

        This is an early warning for DegEq: the system is oscillating
        between competing configurations without converging, analogous
        to the magnetic rotors switching between metastable states.
        """
        return self._hysteresis_active

    def get_cycle_count(self) -> int:
        """Number of detected cycles."""
        return len(self._cycles)

    def get_loop_area(self) -> float:
        """
        Current loop area EMA (energy dissipation proxy).

        Higher values → more energy trapped in oscillation → higher DegEq risk.
        """
        return self._loop_area_ema

    def report(self) -> str:
        """Human-readable hysteresis report."""
        lines = [
            "═══ Hysteresis Analysis ═══",
            f"  Steps monitored : {self._step_count}",
            f"  Peaks detected  : {len(self._rdc_peaks)}",
            f"  Troughs detected: {len(self._rdc_troughs)}",
            f"  Cycles found    : {len(self._cycles)}",
            f"  Hysteresis active: {'YES ⚠' if self._hysteresis_active else 'no'}",
            f"  Loop area (EMA) : {self._loop_area_ema:.4f}",
        ]
        if self._cycles:
            last = self._cycles[-1]
            lines.extend([
                f"  Last cycle:",
                f"    type      : {last['type']}",
                f"    amplitude : {last['amplitude']:.4f}",
                f"    period    : {last['period']} steps",
                f"    loop area : {last['loop_area']:.4f}",
            ])
        lines.append("═══════════════════════════")
        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics dict for logging integration."""
        return {
            "hysteresis_active": float(self._hysteresis_active),
            "cycle_count": len(self._cycles),
            "loop_area_ema": self._loop_area_ema,
            "n_peaks": len(self._rdc_peaks),
            "n_troughs": len(self._rdc_troughs),
            "latest_amplitude": self._cycles[-1]["amplitude"] if self._cycles else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase Trajectory Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class PhaseTrajectoryAnalyzer:
    """
    Analyzes Kuramoto phase trajectories for signatures of frustrated dynamics.

    From Gu et al., the magnetic rotors in the frustrated regime show:
    1. Incomplete rotations (swinging back and forth)
    2. Sublattice-dependent rotation patterns
    3. Staggered rotation directions

    In the Kuramoto context, these correspond to:
    1. Phase oscillation without synchronization (low order parameter)
    2. Sublattice order parameters diverging from global
    3. Anti-correlated phase velocities between heads

    This analyzer detects these patterns and maps them to the
    three collective actuation regimes identified in prior work on
    magnetic rotor arrays:
    - Stripe pattern: alternating alignment → hierarchical head specialization
    - Quarter pattern: quadrant-wise rotation → partial synchronization
    - Staggered pattern: alternating rotation → frustrated dynamics (DegEq risk)

    Args:
        num_heads: Number of attention heads.
        history_length: Number of phase snapshots to retain.
    """

    def __init__(self, num_heads: int, history_length: int = 50):
        self.num_heads = num_heads
        self.history_length = history_length
        self._phase_history: deque = deque(maxlen=history_length)

    def record(self, phases: torch.Tensor) -> None:
        """
        Record a phase snapshot.

        Args:
            phases: [H] or [B, H] phase values (takes mean over batch).
        """
        if phases.dim() > 1:
            phases = phases.mean(dim=0)
        self._phase_history.append(phases.detach().cpu().clone())

    def compute_phase_velocities(self) -> Optional[torch.Tensor]:
        """
        Compute phase velocities dθ/dt from consecutive snapshots.

        Returns:
            velocities: [T-1, H] phase velocities, or None if insufficient data.
        """
        if len(self._phase_history) < 2:
            return None

        phases = torch.stack(list(self._phase_history))  # [T, H]
        dtheta = phases[1:] - phases[:-1]
        # Unwrap: correct for 2π jumps
        dtheta = torch.remainder(dtheta + math.pi, 2 * math.pi) - math.pi
        return dtheta

    def detect_rotation_pattern(self) -> Dict[str, float]:
        """
        Classify the current rotation pattern.

        Returns:
            Dict with:
                pattern: 'synchronized', 'stripe', 'staggered', or 'chaotic'
                velocity_correlation: mean pairwise velocity correlation
                sublattice_coherence: coherence between even/odd head groups
                frustration_score: estimated frustration from phase dynamics
        """
        velocities = self.compute_phase_velocities()
        if velocities is None or velocities.shape[0] < 5:
            return {
                "pattern": "insufficient_data",
                "velocity_correlation": 0.0,
                "sublattice_coherence": 0.0,
                "frustration_score": 0.0,
            }

        H = self.num_heads
        v_recent = velocities[-10:]  # Last 10 steps

        # 1. Mean pairwise velocity correlation
        v_mean = v_recent.mean(dim=0)  # [H]
        v_centered = v_recent - v_mean.unsqueeze(0)
        if v_centered.shape[0] > 1:
            corr_matrix = torch.corrcoef(v_centered.T)
            # Mean off-diagonal correlation
            mask = ~torch.eye(H, dtype=torch.bool)
            vel_corr = corr_matrix[mask].mean().item()
        else:
            vel_corr = 0.0

        # 2. Sublattice coherence (even vs odd heads)
        even_vel = v_recent[:, 0::2].mean(dim=1)  # [T]
        odd_vel = v_recent[:, 1::2].mean(dim=1)   # [T]
        if even_vel.std() > 1e-8 and odd_vel.std() > 1e-8:
            sub_corr = torch.corrcoef(torch.stack([even_vel, odd_vel]))[0, 1].item()
        else:
            sub_corr = 0.0

        # 3. Classify pattern
        if vel_corr > 0.5:
            pattern = "synchronized"    # All heads move together (ferro)
        elif vel_corr < -0.3:
            pattern = "staggered"       # Heads move in opposition (frustrated)
        elif sub_corr < -0.3:
            pattern = "stripe"          # Sublattice-level opposition
        else:
            pattern = "chaotic"         # No clear pattern

        # 4. Frustration score
        # High when sublattices are anti-correlated and velocities are large
        mean_speed = v_recent.abs().mean().item()
        frustration = max(0.0, -sub_corr) * mean_speed

        return {
            "pattern": pattern,
            "velocity_correlation": round(vel_corr, 4),
            "sublattice_coherence": round(sub_corr, 4),
            "frustration_score": round(frustration, 4),
            "mean_angular_speed": round(mean_speed, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Magnetic Friction Diagnostic (comprehensive)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagneticFrictionReport:
    """
    Report from magnetic friction-inspired DegEq analysis.

    Extends the existing DegEqReport (from cgt.diagnostics.degeq) with
    friction-specific metrics.
    """
    # Hysteresis metrics
    hysteresis_active: bool = False
    cycle_count: int = 0
    loop_area_ema: float = 0.0
    latest_amplitude: float = 0.0

    # Phase trajectory metrics
    rotation_pattern: str = "unknown"
    velocity_correlation: float = 0.0
    sublattice_coherence: float = 0.0
    frustration_score: float = 0.0

    # Friction regime classification
    regime: str = "unknown"
    # 'sub-critical': below friction peak → safe
    # 'critical': near friction peak → DegEq imminent
    # 'super-critical': above friction peak → DegEq active
    # 'post-retreat': system has retreated from peak → recovering

    # Risk assessment
    degeq_risk: str = "UNKNOWN"
    interpretation: str = ""

    raw: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "═══ Magnetic Friction DegEq Analysis ═══",
            f"  Hysteresis     : {'ACTIVE ⚠' if self.hysteresis_active else 'inactive'}",
            f"  Cycles detected: {self.cycle_count}",
            f"  Loop area (EMA): {self.loop_area_ema:.4f}",
            f"  Rotation pattern: {self.rotation_pattern}",
            f"  Vel. correlation: {self.velocity_correlation:.4f}",
            f"  Sub. coherence  : {self.sublattice_coherence:.4f}",
            f"  Frustration     : {self.frustration_score:.4f}",
            f"  Friction regime : {self.regime}",
            f"  DegEq risk      : {self.degeq_risk}",
            f"  → {self.interpretation}",
            "═════════════════════════════════════════",
        ]
        return "\n".join(lines)


class MagneticFrictionDiagnostic:
    """
    Comprehensive DegEq diagnostic using magnetic friction-inspired analysis.

    Combines:
    1. HysteresisDetector: cyclic pattern detection in RDC trajectory
    2. PhaseTrajectoryAnalyzer: rotation pattern classification
    3. Friction regime classification

    The friction regime is determined by mapping the observed dynamics
    to the three regimes in Gu et al.:
    - Low coupling (far separation): minimal friction, minimal hierarchy
    - Intermediate coupling (frustration peak): maximal friction → DegEq
    - High coupling (close separation): strong coupling, possible collapse

    Usage:
        diagnostic = MagneticFrictionDiagnostic(
            hysteresis_detector=detector,
            phase_analyzer=analyzer,
        )
        report = diagnostic.run(
            rdc_star=current_rdc_star,
            frustration=current_frustration,
        )
        print(report.summary())
    """

    def __init__(
        self,
        hysteresis_detector: Optional[HysteresisDetector] = None,
        phase_analyzer: Optional[PhaseTrajectoryAnalyzer] = None,
    ):
        self.hysteresis = hysteresis_detector
        self.phase_analyzer = phase_analyzer

    def run(
        self,
        rdc_star: Optional[float] = None,
        frustration: Optional[float] = None,
        order_parameter: Optional[float] = None,
    ) -> MagneticFrictionReport:
        """
        Run full magnetic friction diagnostic.

        Args:
            rdc_star: Current RDC attractor value (from DegEqDiagnostics).
            frustration: Current frustration parameter F.
            order_parameter: Current Kuramoto order parameter r.

        Returns:
            MagneticFrictionReport with all computed fields.
        """
        report = MagneticFrictionReport()

        # 1. Hysteresis analysis
        if self.hysteresis is not None:
            report.hysteresis_active = self.hysteresis.hysteresis_detected()
            report.cycle_count = self.hysteresis.get_cycle_count()
            report.loop_area_ema = self.hysteresis.get_loop_area()
            metrics = self.hysteresis.get_metrics()
            report.latest_amplitude = metrics.get("latest_amplitude", 0.0)
            report.raw["hysteresis"] = metrics

        # 2. Phase trajectory analysis
        if self.phase_analyzer is not None:
            pattern_info = self.phase_analyzer.detect_rotation_pattern()
            report.rotation_pattern = pattern_info["pattern"]
            report.velocity_correlation = pattern_info["velocity_correlation"]
            report.sublattice_coherence = pattern_info["sublattice_coherence"]
            report.frustration_score = pattern_info["frustration_score"]
            report.raw["phase_trajectory"] = pattern_info

        # 3. Friction regime classification
        report.regime = self._classify_regime(
            rdc_star=rdc_star,
            frustration=frustration,
            hysteresis=report.hysteresis_active,
            pattern=report.rotation_pattern,
        )

        # 4. Risk assessment
        report.degeq_risk = self._assess_risk(report)
        report.interpretation = self._interpret(report)

        return report

    def _classify_regime(
        self,
        rdc_star: Optional[float],
        frustration: Optional[float],
        hysteresis: bool,
        pattern: str,
    ) -> str:
        """
        Classify current friction regime.

        Maps to Gu et al. Fig. 4:
        - sub-critical: far from friction peak, safe training
        - critical: at or near friction peak, DegEq forming
        - super-critical: past peak, DegEq active, radial structure collapsed
        - post-retreat: system has been moved away from peak
        """
        # Primary indicator: rdc_star
        if rdc_star is not None:
            if rdc_star > 8.0:
                return "super-critical"
            elif rdc_star > 4.0:
                return "critical"
            elif rdc_star < 2.0:
                return "sub-critical"

        # Secondary: frustration + hysteresis
        if frustration is not None:
            if frustration > 0.6 and hysteresis:
                return "critical"
            elif frustration > 0.7:
                return "super-critical"
            elif frustration < 0.2:
                return "sub-critical"

        # Tertiary: rotation pattern
        if pattern == "staggered":
            return "critical"
        elif pattern == "synchronized":
            return "sub-critical"

        return "unknown"

    def _assess_risk(self, report: MagneticFrictionReport) -> str:
        """Assess DegEq risk level."""
        risk_score = 0

        if report.regime == "super-critical":
            risk_score += 3
        elif report.regime == "critical":
            risk_score += 2

        if report.hysteresis_active:
            risk_score += 2

        if report.rotation_pattern == "staggered":
            risk_score += 1

        if report.frustration_score > 0.5:
            risk_score += 1

        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _interpret(self, report: MagneticFrictionReport) -> str:
        """Generate human-readable interpretation."""
        parts = []

        if report.regime == "super-critical":
            parts.append(
                "System is PAST the friction peak — DegEq attractor is active. "
                "Radial structure is likely collapsed."
            )
        elif report.regime == "critical":
            parts.append(
                "System is NEAR the friction peak — DegEq is forming. "
                "Consider reducing coupling strength or activating AngularLMHead."
            )
        elif report.regime == "sub-critical":
            parts.append(
                "System is BELOW the friction peak — training dynamics are healthy."
            )

        if report.hysteresis_active:
            parts.append(
                f"Hysteretic cycling detected ({report.cycle_count} cycles, "
                f"amplitude {report.latest_amplitude:.3f}). "
                "This indicates the system is oscillating between competing "
                "configurations without convergence — a direct analog of the "
                "magnetic rotor frustrated switching in Gu et al."
            )

        if report.rotation_pattern == "staggered":
            parts.append(
                "Phase dynamics show staggered rotation pattern (anti-correlated "
                "sublattice velocities), consistent with frustrated magnetic ordering."
            )

        return " ".join(parts) if parts else "Insufficient data for interpretation."
