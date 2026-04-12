# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Magnetic Friction Integration
==============================

Unified integration layer for all magnetic friction-inspired components.

This module provides:
    1. MagneticFrictionConfig: configuration dataclass for all new components
    2. MagneticFrictionExtension: mixin/wrapper that adds friction-aware
       capabilities to existing H-AKORN training without modifying any
       existing module
    3. Training loop hooks: step(), log(), report()

Design: PURELY ADDITIVE. This module wraps existing components and adds
new functionality alongside them. No existing import is replaced.

Architecture:
    Existing (untouched)                 New (this extension)
    ─────────────────────                ─────────────────────
    AdaptiveCoupling          ←wrapped→  FrustrationAwareCoupling
    DynamicCouplingScheduler  ←replaced→ FrictionAwareCouplingScheduler
    HyperbolicLMHeadV2        ←extended→ SublatticeLMHead
    DegEqDiagnostics          ←extended→ MagneticFrictionDiagnostic
                                         HysteresisDetector
                                         PhaseTrajectoryAnalyzer

Usage in training loop:
    from cgt.integration.magnetic_friction import (
        MagneticFrictionConfig,
        MagneticFrictionExtension,
    )

    config = MagneticFrictionConfig(
        enable_frustration_coupling=True,
        enable_sublattice_lm_head=True,
        enable_hysteresis_detection=True,
    )
    ext = MagneticFrictionExtension(config, num_heads=4)

    for step in range(total_steps):
        # ... existing training step ...

        # Hook: update friction-aware components
        ext.on_step(
            rdc_ema=rdc_ema,
            phases=current_phases,
            logit_std=logit_std,
            l_hidden=l_hidden,
        )

        # Periodic logging
        if step % log_interval == 0:
            metrics = ext.get_metrics()
            # → includes frustration, hysteresis, friction regime, etc.

    # Post-training
    report = ext.final_report()
    print(report)

References:
    Gu, Lüders & Bechinger (2026). Nature Materials. DOI:10.1038/s41563-026-02538-1
    de Sena (2026). HyDRA v4: Degenerate Equilibrium in Hyperbolic Distillation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class MagneticFrictionConfig:
    """
    Configuration for magnetic friction extension.

    All flags default to False — opt-in only.
    """
    # ── Feature flags ──
    enable_frustration_coupling: bool = False
    enable_sublattice_lm_head: bool = False
    enable_hysteresis_detection: bool = True   # lightweight, always useful
    enable_phase_analysis: bool = True          # lightweight, always useful
    enable_nonmonotonic_scheduler: bool = False

    # ── FrustrationAwareCoupling params ──
    frustration_mode: str = 'sublattice'
    frustration_strength: float = 0.3
    frustration_ceiling: float = 0.7
    sublattice_split: str = 'even_odd'

    # ── SublatticeLMHead params ──
    r_max_frequent: float = 1.5
    r_max_rare: float = 5.0
    frequency_threshold: float = 0.3
    angular_mode: bool = False    # True → combines with Channel 2 fix

    # ── HysteresisDetector params ──
    hysteresis_window: int = 200
    hysteresis_min_amplitude: float = 0.5

    # ── Scheduler params ──
    scheduler_type: str = 'friction_aware'   # or 'non_monotonic'
    coupling_min: float = 0.1
    coupling_max: float = 2.0
    warmup_steps: int = 500
    frustration_threshold: float = 0.5
    frustration_target: float = 0.25


class MagneticFrictionExtension:
    """
    Mixin that adds magnetic friction-aware capabilities to H-AKORN training.

    Instantiate alongside existing training components. Call on_step() in the
    training loop. All existing code remains untouched.

    This class does NOT own any model parameters — it only monitors and
    provides diagnostics/scheduling. The actual model modifications
    (FrustrationAwareCoupling, SublatticeLMHead) are separate modules
    that must be instantiated and used directly in the model architecture.
    """

    def __init__(
        self,
        config: MagneticFrictionConfig,
        num_heads: int = 4,
    ):
        self.config = config
        self.num_heads = num_heads

        # ── Hysteresis detector (always lightweight) ──
        if config.enable_hysteresis_detection:
            from cgt.diagnostics.hysteresis import HysteresisDetector
            self.hysteresis = HysteresisDetector(
                window_size=config.hysteresis_window,
                min_amplitude=config.hysteresis_min_amplitude,
            )
        else:
            self.hysteresis = None

        # ── Phase trajectory analyzer ──
        if config.enable_phase_analysis:
            from cgt.diagnostics.hysteresis import PhaseTrajectoryAnalyzer
            self.phase_analyzer = PhaseTrajectoryAnalyzer(
                num_heads=num_heads,
            )
        else:
            self.phase_analyzer = None

        # ── Friction-aware scheduler ──
        if config.enable_nonmonotonic_scheduler:
            from cgt.dynamics.magnetic_friction_scheduler import (
                FrictionAwareCouplingScheduler,
                NonMonotonicCouplingScheduler,
            )
            if config.scheduler_type == 'friction_aware':
                self.scheduler = FrictionAwareCouplingScheduler(
                    coupling_min=config.coupling_min,
                    coupling_max=config.coupling_max,
                    warmup_steps=config.warmup_steps,
                    frustration_threshold=config.frustration_threshold,
                    frustration_target=config.frustration_target,
                )
            else:
                self.scheduler = NonMonotonicCouplingScheduler(
                    coupling_min=config.coupling_min,
                    coupling_max=config.coupling_max,
                    warmup_steps=config.warmup_steps,
                )
        else:
            self.scheduler = None

        # ── Step counter ──
        self._step = 0

        # ── Metric history for plotting ──
        self._metric_history: List[Dict[str, float]] = []

    def on_step(
        self,
        rdc_ema: Optional[float] = None,
        phases: Optional[torch.Tensor] = None,
        logit_std: Optional[float] = None,
        l_hidden: Optional[float] = None,
        order_parameter: Optional[float] = None,
        frustration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Training loop hook. Call after each training step.

        Args:
            rdc_ema: Current RDC EMA value.
            phases: [B, H] or [H] current Kuramoto phases.
            logit_std: Current logit standard deviation.
            l_hidden: Current hidden alignment loss.
            order_parameter: Kuramoto order parameter r.
            frustration: Pre-computed frustration (from FrustrationAwareCoupling).

        Returns:
            Dict with any alerts or scheduling updates.
        """
        result: Dict[str, Any] = {"step": self._step}

        # 1. Update hysteresis detector
        if self.hysteresis is not None and rdc_ema is not None:
            self.hysteresis.update(
                rdc_ema=rdc_ema,
                order_parameter=order_parameter,
                logit_std=logit_std,
                l_hidden=l_hidden,
            )
            if self.hysteresis.hysteresis_detected():
                result["alert"] = "hysteresis_detected"
                result["loop_area"] = self.hysteresis.get_loop_area()

        # 2. Record phase trajectory
        if self.phase_analyzer is not None and phases is not None:
            self.phase_analyzer.record(phases)

        # 3. Update scheduler
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'get_coupling'):
                coupling = self.scheduler.get_coupling(
                    phases=phases,
                    frustration=frustration,
                )
                result["scheduled_coupling"] = coupling
                self.scheduler.step()
            elif hasattr(self.scheduler, 'forward'):
                coupling = self.scheduler()
                result["scheduled_coupling"] = coupling
                self.scheduler.step()

        self._step += 1
        return result

    def get_metrics(self) -> Dict[str, float]:
        """
        Collect all current metrics for logging.

        Returns:
            Dict with all magnetic friction metrics.
        """
        metrics: Dict[str, float] = {"mf_step": self._step}

        if self.hysteresis is not None:
            hyst = self.hysteresis.get_metrics()
            for k, v in hyst.items():
                metrics[f"mf_hyst_{k}"] = v

        if self.phase_analyzer is not None:
            phase = self.phase_analyzer.detect_rotation_pattern()
            metrics["mf_vel_corr"] = phase["velocity_correlation"]
            metrics["mf_sub_coherence"] = phase["sublattice_coherence"]
            metrics["mf_frustration_score"] = phase["frustration_score"]
            metrics["mf_pattern"] = {
                "synchronized": 0.0,
                "stripe": 1.0,
                "staggered": 2.0,
                "chaotic": 3.0,
            }.get(phase["pattern"], -1.0)

        if self.scheduler is not None:
            if hasattr(self.scheduler, 'get_state'):
                sched = self.scheduler.get_state()
                metrics["mf_coupling"] = sched.get("coupling", 0.0)
                metrics["mf_frustration_ema"] = sched.get("frustration_ema", 0.0)

        return metrics

    def final_report(
        self,
        rdc_star: Optional[float] = None,
    ) -> str:
        """
        Generate comprehensive post-training report.

        Args:
            rdc_star: Final RDC attractor value.

        Returns:
            Human-readable report string.
        """
        from cgt.diagnostics.hysteresis import (
            MagneticFrictionDiagnostic,
        )

        diag = MagneticFrictionDiagnostic(
            hysteresis_detector=self.hysteresis,
            phase_analyzer=self.phase_analyzer,
        )

        # Extract frustration from latest metrics
        frustration = None
        if self.phase_analyzer is not None:
            pattern = self.phase_analyzer.detect_rotation_pattern()
            frustration = pattern.get("frustration_score")

        report = diag.run(
            rdc_star=rdc_star,
            frustration=frustration,
        )

        return report.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions for model construction
# ─────────────────────────────────────────────────────────────────────────────

def create_frustration_coupling(
    config: MagneticFrictionConfig,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """
    Factory: create FrustrationAwareCoupling from config.

    Use this instead of AdaptiveCoupling when frustration-aware training
    is desired. The returned module has the same forward() signature.

    Args:
        config: MagneticFrictionConfig instance.
        num_heads: Number of attention heads.
        **kwargs: Additional kwargs passed to FrustrationAwareCoupling.

    Returns:
        FrustrationAwareCoupling module.
    """
    from cgt.models.hakorn.magnetic_coupling import FrustrationAwareCoupling

    return FrustrationAwareCoupling(
        num_heads=num_heads,
        frustration_mode=config.frustration_mode,
        frustration_strength=config.frustration_strength,
        frustration_ceiling=config.frustration_ceiling,
        sublattice_split=config.sublattice_split,
        **kwargs,
    )


def create_sublattice_lm_head(
    config: MagneticFrictionConfig,
    n_embd: int,
    vocab_size: int,
    substrate,
    token_counts: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory: create SublatticeLMHead from config.

    Use this instead of HyperbolicLMHeadV2 when sublattice-stratified
    radial bounds are desired.

    Args:
        config: MagneticFrictionConfig instance.
        n_embd: Embedding dimension.
        vocab_size: Vocabulary size.
        substrate: LorentzSubstrateV2 instance.
        token_counts: [V] token occurrence counts.
        **kwargs: Additional kwargs passed to SublatticeLMHead.

    Returns:
        SublatticeLMHead module.
    """
    from cgt.models.sublattice_lm_head import SublatticeLMHead

    return SublatticeLMHead(
        n_embd=n_embd,
        vocab_size=vocab_size,
        substrate=substrate,
        token_counts=token_counts,
        r_max_frequent=config.r_max_frequent,
        r_max_rare=config.r_max_rare,
        frequency_threshold=config.frequency_threshold,
        angular_mode=config.angular_mode,
        **kwargs,
    )
