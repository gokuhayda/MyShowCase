# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
"""
adaptive_controller.py
======================
AdaptiveHyperController — closed-loop hyperparameter control for HyDRA distillation.

Architecture
------------
Three independent control levels run at different timescales:

  Level 1 — GradBalance controller  (every grad_balance_every steps)
      Proportional controller that keeps eff_kl ≈ eff_hidden.
      Adjusts lambda_distill and lambda_hidden.
      Target: imbalance < balance_tolerance (default 2.0x).

  Level 2 — PPL velocity controller  (every ppl_window steps)
      Monitors val_ppl improvement rate against an exponential target.
      Reduces LR if improvement stalls; increases if faster than expected.
      Based on cosine-annealing with restarts (SGDR, Loshchilov 2017).

  Level 3 — Regime controller  (every regime_every steps)
      Monitors rdc, logit_std, div for pathological regimes.
      Replaces the heuristic AutoTune with metric-based decisions.
      Detects: DegEq onset, RadiusCollapse risk, OverconfidenceSpiral.

Population Based Training (PBT) — multi-GPU extension
------------------------------------------------------
When n_workers > 1, PBT mode activates. Each worker runs independently
with perturbed hyperparameters. Every pbt_interval steps:
  1. All workers synchronise val_ppl via a shared file or torch.distributed.
  2. Bottom 20% workers copy weights from top 20% (exploit).
  3. Copied workers perturb their hyperparameters (explore).
  4. Training resumes with new hyperparameters.

This matches the DeepMind PBT paper (Jaderberg et al. 2017) with the
hyperbolic-specific modification that geometry metrics (rdc, rad) are
included in the fitness function alongside val_ppl.

Single-GPU mode (default)
--------------------------
PBT is disabled. Only Levels 1–3 run. No inter-process communication.
All state is contained within the controller instance.

Usage
-----
    from cgt.distillation.adaptive_controller import AdaptiveHyperController

    controller = AdaptiveHyperController(config=dist_cfg, trainer=trainer)

    # In training loop (called automatically if wired via trainer hook):
    controller.step(metrics=m, step=trainer.step)

    # Or wire to trainer:
    trainer.adaptive_controller = controller
"""

from __future__ import annotations

import math
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch




# ─────────────────────────────────────────────────────────────────────────────
# GradNorm Measure — V3 (adapted from CGTGradNormMeasure, Chen et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class GradNormMeasure:
    """
    Measures actual per-loss gradient norms via autograd.grad on a probe parameter.

    V3 of the adaptive controller uses ‖∇_probe(λ_i · L_i)‖₂ instead of
    abs(loss_magnitude) for gradient balance decisions.

    The probe is the first LayerNorm weight found in the student model —
    small enough for fast computation, deep enough to receive gradients
    from all loss terms.

    Cost: ~N extra autograd.grad calls (retain_graph=True) every
    measure_every steps, where N = number of loss components.
    At measure_every=500: negligible overhead (<1%).

    Reference: Chen et al. GradNorm (2018), adapted by Sena (2026).
    """

    def __init__(self, measure_every: int = 500):
        self.measure_every = measure_every
        self._probe_cache: Optional[torch.nn.Parameter] = None

    def _find_probe(self, model: "torch.nn.Module") -> "Optional[torch.nn.Parameter]":
        """Find a small parameter deep in the model as gradient probe."""
        if self._probe_cache is not None:
            return self._probe_cache
        # Priority 1: LayerNorm weight (small, receives all gradients)
        for name, p in model.named_parameters():
            if 'norm' in name.lower() and 'weight' in name and p.requires_grad and p.numel() < 512:
                self._probe_cache = p
                return p
        # Priority 2: any small weight
        for name, p in model.named_parameters():
            if 'weight' in name and p.requires_grad and p.dim() >= 1 and p.numel() < 256:
                self._probe_cache = p
                return p
        # Fallback: first trainable parameter
        for p in model.parameters():
            if p.requires_grad:
                self._probe_cache = p
                return p
        return None

    def measure(
        self,
        loss_components: "Dict[str, torch.Tensor]",
        model:           "torch.nn.Module",
        step:            int,
    ) -> "Dict[str, float]":
        """
        Measure ‖∇_probe(L_i)‖₂ for each loss component.

        Args:
            loss_components: {name: loss_tensor} — must have valid computation graph
            model:           student model (to find probe parameter)
            step:            current training step

        Returns:
            {name: grad_norm, 'imbalance_ratio': float}
            Empty dict if not a measurement step or probe not found.
        """
        if step % self.measure_every != 0:
            return {}

        probe = self._find_probe(model)
        if probe is None:
            return {}

        norms: Dict[str, float] = {}
        for name, loss_val in loss_components.items():
            if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
                norms[name] = 0.0
                continue
            try:
                grad = torch.autograd.grad(
                    loss_val, probe,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                norms[name] = float(grad.norm().item()) if grad is not None else 0.0
            except Exception:
                norms[name] = 0.0

        active = {k: v for k, v in norms.items() if v > 1e-12}
        if len(active) >= 2:
            norms['imbalance_ratio'] = max(active.values()) / min(active.values())
        else:
            norms['imbalance_ratio'] = 1.0

        return norms

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveControllerConfig:
    """
    Configuration for AdaptiveHyperController.

    Single-GPU defaults are conservative — small adjustments, wide guards.
    Multi-GPU PBT defaults follow DeepMind (2017): 20% exploit/explore,
    perturbation factor 0.8/1.2.
    """

    # ── Level 1: GradBalance ─────────────────────────────────────────────────
    grad_balance_every:  int   = 500    # steps between balance checks (CGT: 50 batches)
    balance_tolerance:   float = 2.0    # max allowed eff_kl/eff_hidden ratio
    balance_lr:          float = 0.5    # how aggressively to correct (0–1)
    lambda_distill_min:  float = 0.001  # lower bound
    lambda_distill_max:  float = 0.5    # upper bound
    lambda_hidden_min:   float = 0.001  # lower bound
    lambda_hidden_max:   float = 0.3    # upper bound

    # ── Level 2: PPL velocity ────────────────────────────────────────────────
    ppl_window:          int   = 3      # evals to average for velocity
    ppl_stall_patience:  int   = 5      # evals below min_velocity before LR drop
    ppl_min_velocity:    float = 0.005  # min relative PPL improvement per eval
    lr_reduce_factor:    float = 0.7    # LR multiplier on stall
    lr_boost_factor:     float = 1.1    # LR multiplier on fast convergence
    lr_min:              float = 1e-6   # hard lower bound
    lr_max:              float = 5e-4   # hard upper bound
    warmup_guard:        int   = 5000   # no LR changes before this step

    # ── Level 3: Regime ──────────────────────────────────────────────────────
    regime_every:        int   = 200    # steps between regime checks
    rdc_danger:          float = 3.0    # rdc above this = DegEq onset
    rdc_critical:        float = 7.0    # rdc above this = DegEq confirmed
    logit_std_max:       float = 15.0   # above this = OverconfidenceSpiral
    div_min:             float = 0.3    # below this = DiversityCollapse
    rad_min:             float = 0.05   # below this = RadiusCollapse risk
    rad_max:             float = 4.0    # above this = RadiusDrift (DegEq precursor)
    loss_ema_beta:       float = 0.95   # EMA for loss magnitude tracking (CGT method)

    # ── V3: GradNorm real (Chen et al. 2018) ────────────────────────────────
    use_gradnorm:        bool  = True   # V3: use ‖∇L_i‖₂ instead of loss magnitude
    gradnorm_alpha:      float = 0.5    # GradNorm correction strength (0=off, 1=full)
    gradnorm_clamp:      float = 2.0    # max scale factor per step (prevents oscillation)
    regime_confirm:      int   = 3      # consecutive bad checks before action

    # ── PBT: Population Based Training (multi-GPU) ───────────────────────────
    n_workers:           int   = 1      # 1 = single GPU (PBT disabled)
    pbt_interval:        int   = 10000  # steps between PBT synchronisation
    pbt_exploit_frac:    float = 0.2    # fraction of workers to replace
    pbt_perturb_factor:  float = 0.2    # perturbation magnitude (±20%)
    pbt_fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "val_ppl":  0.5,   # lower is better
        "rdc_ema":  0.3,   # lower is better
        "logit_std": 0.2,  # target range 1–5
    })
    pbt_shared_dir:      Optional[str] = None  # shared filesystem for PBT state
    worker_id:           int   = 0      # this worker's rank (0-indexed)

    # ── General ──────────────────────────────────────────────────────────────
    enabled:             bool  = True   # master switch
    verbose:             bool  = True   # print actions
    log_to_file:         bool  = True   # write controller log to CSV
    log_path:            Optional[str] = None  # None = auto-detect from trainer


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveHyperController:
    """
    Closed-loop hyperparameter controller for HyDRA distillation.

    Wires into DistillationTrainerV2 via:
        trainer.adaptive_controller = AdaptiveHyperController(cfg, trainer)

    The controller's .step() is called automatically inside the trainer's
    train loop at each log_every step if the trainer detects the attribute.

    Alternatively, call .step(metrics, step) manually after each
    distillation_step().
    """

    def __init__(
        self,
        config: AdaptiveControllerConfig,
        trainer,   # DistillationTrainerV2
    ):
        self.cfg     = config
        self.trainer = trainer
        self.dcfg    = trainer.config   # DistillationConfigV2

        # ── Internal state ────────────────────────────────────────────────────
        # V3: GradNorm measure
        self._grad_measure = GradNormMeasure(measure_every=config.grad_balance_every)
        self._grad_norm_ema: Dict[str, float] = {}  # EMA of gradient norms
        self._ppl_history:    List[float] = []
        self._stall_count:    int         = 0
        self._regime_counts:  Dict[str, int] = {
            "degeq": 0, "overconfidence": 0,
            "diversity": 0, "radius": 0, "radius_drift": 0,
        }
        # Loss magnitude EMA (CGT method — faster than GradBalance)
        self._loss_ema: Dict[str, float] = {}
        self._action_log:     List[Dict]  = []
        self._last_gb_step:   int         = 0
        self._last_pbt_step:  int         = 0

        # ── PBT state ─────────────────────────────────────────────────────────
        self._pbt_active = config.n_workers > 1
        if self._pbt_active and config.pbt_shared_dir is None:
            warnings.warn(
                "AdaptiveHyperController: PBT enabled but pbt_shared_dir is None. "
                "Workers cannot synchronise. Set pbt_shared_dir to a shared path "
                "(e.g. Google Drive mount point).",
                UserWarning
            )

        if config.verbose:
            mode = f"PBT ({config.n_workers} workers)" if self._pbt_active else "single-GPU"
            print(f"  [AdaptiveController] Initialised — mode={mode}  "
                  f"L1@{config.grad_balance_every}  "
                  f"L2@every_eval  "
                  f"L3@{config.regime_every}")

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, metrics: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Main entry point. Call after each distillation_step().

        Args:
            metrics: dict returned by distillation_step()
            step:    current training step

        Returns:
            dict of actions taken (empty if none)
        """
        if not self.cfg.enabled:
            return {}

        actions = {}

        # Level 3 runs most frequently — every regime_every steps
        if step % self.cfg.regime_every == 0:
            actions.update(self._level3_regime(metrics, step))

        # Level 1 runs at grad_balance_every
        if step - self._last_gb_step >= self.cfg.grad_balance_every:
            gb = self._get_grad_balance()
            if gb:
                actions.update(self._level1_grad_balance(gb, step))
            self._last_gb_step = step

        # PBT synchronisation
        if self._pbt_active and step - self._last_pbt_step >= self.cfg.pbt_interval:
            actions.update(self._pbt_sync(step))
            self._last_pbt_step = step

        if actions:
            self._action_log.append({"step": step, **actions})
            if self.cfg.log_to_file:
                self._write_log()

        return actions

    def on_eval(self, val_ppl: float, step: int) -> Dict[str, Any]:
        """
        Call after each validation step with the val_ppl result.
        Runs Level 2 PPL velocity control.
        """
        if not self.cfg.enabled:
            return {}
        return self._level2_ppl_velocity(val_ppl, step)

    def get_state(self) -> Dict:
        """Serialise controller state for checkpointing."""
        return {
            "ppl_history":   self._ppl_history,
            "stall_count":   self._stall_count,
            "regime_counts": self._regime_counts,
            "last_gb_step":  self._last_gb_step,
            "lambda_distill": self.dcfg.lambda_distill,
            "lambda_hidden":  self.dcfg.lambda_hidden,
        }

    def load_state(self, state: Dict) -> None:
        """Restore controller state from checkpoint."""
        self._ppl_history   = state.get("ppl_history",   [])
        self._stall_count   = state.get("stall_count",   0)
        self._regime_counts = state.get("regime_counts", self._regime_counts)
        self._last_gb_step  = state.get("last_gb_step",  0)

    # ── Level 1: GradBalance ─────────────────────────────────────────────────

    def _level1_grad_balance(
        self, gb: Dict[str, float], step: int
    ) -> Dict[str, Any]:
        """
        V1/V2: Proportional controller using eff_kl / eff_hidden ratio.
        V3: Uses real gradient norms ‖∇L_i‖₂ when use_gradnorm=True.

        V3 algorithm (GradNorm, Chen et al. 2018):
            scale = (target_norm / current_norm) ^ gradnorm_alpha
            scale = clamp(scale, 1/gradnorm_clamp, gradnorm_clamp)
            lambda_i *= scale
        """
        # ── V3: Try GradNorm first ────────────────────────────────────────
        if self.cfg.use_gradnorm and hasattr(self, '_grad_measure'):
            student = getattr(self.trainer, 'student', None)
            if student is not None:
                # Build loss component tensors from trainer's last step
                loss_components = self._get_loss_components()
                if loss_components:
                    gnorms = self._grad_measure.measure(loss_components, student, step)
                    if gnorms and 'imbalance_ratio' in gnorms:
                        return self._level1_gradnorm_v3(gnorms, step)

        # ── V1/V2 fallback: use GradBalance report ────────────────────────
        eff_kl  = gb.get("eff_kl",  0.0)
        eff_hid = gb.get("eff_hid", 0.0)

        if eff_kl < 1e-6 or eff_hid < 1e-6:
            return {}

        ratio = max(eff_kl, eff_hid) / min(eff_kl, eff_hid)
        if ratio <= self.cfg.balance_tolerance:
            return {}

        actions = {}
        lr = self.cfg.balance_lr

        if eff_kl > eff_hid:
            correction = 1.0 - lr * (ratio - 1.0) / ratio
            old = self.dcfg.lambda_distill
            new = float(max(
                self.cfg.lambda_distill_min,
                min(self.cfg.lambda_distill_max, old * correction)
            ))
            self.dcfg.lambda_distill = new
            if hasattr(self.trainer, 'config'):
                self.trainer.config.lambda_distill = new
            actions["L1_lambda_distill"] = f"{old:.4f} → {new:.4f} (V1/V2 fallback)"
            if self.cfg.verbose:
                print(f"  [L1-V2] KL dominates ({ratio:.1f}x) "
                      f"→ lambda_distill: {old:.4f} → {new:.4f}")
        else:
            correction = 1.0 - lr * (ratio - 1.0) / ratio
            old = self.dcfg.lambda_hidden
            new = float(max(
                self.cfg.lambda_hidden_min,
                min(self.cfg.lambda_hidden_max, old * correction)
            ))
            self.dcfg.lambda_hidden = new
            if hasattr(self.trainer, 'config'):
                self.trainer.config.lambda_hidden = new
            actions["L1_lambda_hidden"] = f"{old:.4f} → {new:.4f} (V1/V2 fallback)"
            if self.cfg.verbose:
                print(f"  [L1-V2] Hidden dominates ({ratio:.1f}x) "
                      f"→ lambda_hidden: {old:.4f} → {new:.4f}")

        return actions

    def _level1_gradnorm_v3(
        self, gnorms: Dict[str, float], step: int
    ) -> Dict[str, Any]:
        """
        V3 GradNorm correction: scale each lambda by (target/current)^alpha.
        target = mean gradient norm across active components.
        """
        imbalance = gnorms.pop('imbalance_ratio', 1.0)
        if imbalance <= self.cfg.balance_tolerance:
            return {}

        active = {k: v for k, v in gnorms.items() if isinstance(v, float) and v > 1e-12}
        if len(active) < 2:
            return {}

        target = sum(active.values()) / len(active)
        alpha  = self.cfg.gradnorm_alpha
        clamp  = self.cfg.gradnorm_clamp
        actions: Dict[str, Any] = {}

        # Map grad norm keys to lambda attrs
        key_to_lambda = {
            'distill': ('lambda_distill', self.cfg.lambda_distill_min, self.cfg.lambda_distill_max),
            'hidden':  ('lambda_hidden',  self.cfg.lambda_hidden_min,  self.cfg.lambda_hidden_max),
        }

        for key, current_norm in active.items():
            # Normalise key to match lambda name
            lam_key = next((k for k in key_to_lambda if k in key.lower()), None)
            if lam_key is None:
                continue
            attr, lo, hi = key_to_lambda[lam_key]
            old = getattr(self.dcfg, attr, None)
            if old is None:
                continue

            scale = (target / current_norm) ** alpha
            scale = max(1.0 / clamp, min(clamp, scale))
            new   = float(max(lo, min(hi, old * scale)))

            if abs(new - old) / max(old, 1e-8) > 0.02:  # >2% change threshold
                setattr(self.dcfg, attr, new)
                if hasattr(self.trainer, 'config'):
                    setattr(self.trainer.config, attr, new)
                direction = '↓' if new < old else '↑'
                actions[f"L1_V3_{attr}"] = (
                    f"{old:.4f} → {new:.4f} {direction} "
                    f"(‖∇‖={current_norm:.4f}, target={target:.4f}, imb={imbalance:.1f}x)"
                )
                if self.cfg.verbose:
                    print(f"  [L1-V3-GradNorm] {attr}: {old:.4f} → {new:.4f} {direction} "
                          f"(‖∇‖={current_norm:.4f}, target={target:.4f})")

        return actions

    def _get_loss_components(self) -> "Dict[str, torch.Tensor]":
        """
        Extract individual loss tensors from the trainer's last step metrics.
        Returns tensors that still have computation graphs when called mid-step.
        Falls back to empty dict if not available.
        """
        # Try to get from trainer's live loss cache if it exists
        live = getattr(self.trainer, '_live_loss_components', {})
        if live:
            return {k: v for k, v in live.items()
                    if isinstance(v, torch.Tensor) and v.requires_grad}
        return {}

    # ── Level 2: PPL velocity ────────────────────────────────────────────────

    def _level2_ppl_velocity(self, val_ppl: float, step: int) -> Dict[str, Any]:
        """
        Monitors PPL improvement rate and adjusts LR.

        Velocity = relative improvement per eval:
            v = (ppl_prev - ppl_cur) / ppl_prev

        If v < min_velocity for ppl_stall_patience evals → reduce LR.
        If v > 3 × min_velocity consistently → boost LR slightly.
        """
        if step < self.cfg.warmup_guard:
            self._ppl_history.append(val_ppl)
            return {}

        self._ppl_history.append(val_ppl)
        if len(self._ppl_history) < 2:
            return {}

        # Compute velocity over window
        window = self._ppl_history[-self.cfg.ppl_window:]
        if len(window) < 2:
            return {}

        velocity = (window[-2] - window[-1]) / max(window[-2], 1e-6)
        actions  = {}

        opt = self.trainer.optimizer

        if velocity < self.cfg.ppl_min_velocity:
            self._stall_count += 1
            if self._stall_count >= self.cfg.ppl_stall_patience:
                # Reduce LR
                for pg in opt.param_groups:
                    old_lr = pg["lr"]
                    new_lr = max(self.cfg.lr_min,
                                 old_lr * self.cfg.lr_reduce_factor)
                    pg["lr"] = new_lr
                actions["L2_lr_reduce"] = f"{old_lr:.2e} → {new_lr:.2e}"
                self._stall_count = 0
                if self.cfg.verbose:
                    print(f"  [L2-PPLvelocity] Stall detected (v={velocity:.4f}) "
                          f"→ LR: {old_lr:.2e} → {new_lr:.2e}")
        else:
            self._stall_count = max(0, self._stall_count - 1)
            # Optionally boost LR on fast convergence
            if velocity > 3 * self.cfg.ppl_min_velocity:
                for pg in opt.param_groups:
                    old_lr = pg["lr"]
                    new_lr = min(self.cfg.lr_max,
                                 old_lr * self.cfg.lr_boost_factor)
                    pg["lr"] = new_lr
                actions["L2_lr_boost"] = f"{old_lr:.2e} → {new_lr:.2e}"
                if self.cfg.verbose:
                    print(f"  [L2-PPLvelocity] Fast convergence (v={velocity:.4f}) "
                          f"→ LR: {old_lr:.2e} → {new_lr:.2e}")

        return actions

    # ── Level 3: Regime ──────────────────────────────────────────────────────

    def _level3_regime(
        self, metrics: Dict[str, Any], step: int
    ) -> Dict[str, Any]:
        """
        Detects and responds to pathological training regimes.

        Regimes monitored:
          DegEq onset:          rdc > rdc_danger for confirm steps
          OverconfidenceSpiral: logit_std > logit_std_max
          DiversityCollapse:    div < div_min
          RadiusCollapse risk:  mean_radius < rad_min
          RadiusDrift:          mean_radius > rad_max  (DegEq precursor — CGT insight)
        """
        rdc       = float(metrics.get("rdc_proxy",    0.0))
        logit_std = float(metrics.get("logit_std",    1.0))
        div       = float(metrics.get("diversity",    1.0))
        rad       = float(metrics.get("mean_radius",  1.5))

        actions = {}

        # DegEq onset
        if rdc > self.cfg.rdc_danger:
            self._regime_counts["degeq"] += 1
            if self._regime_counts["degeq"] >= self.cfg.regime_confirm:
                action = self._respond_degeq(rdc, step)
                if action:
                    actions["L3_degeq"] = action
                    self._regime_counts["degeq"] = 0
        else:
            self._regime_counts["degeq"] = max(0, self._regime_counts["degeq"] - 1)

        # OverconfidenceSpiral
        if logit_std > self.cfg.logit_std_max:
            self._regime_counts["overconfidence"] += 1
            if self._regime_counts["overconfidence"] >= self.cfg.regime_confirm:
                action = self._respond_overconfidence(logit_std, step)
                if action:
                    actions["L3_overconfidence"] = action
                    self._regime_counts["overconfidence"] = 0
        else:
            self._regime_counts["overconfidence"] = 0

        # DiversityCollapse
        if div < self.cfg.div_min:
            self._regime_counts["diversity"] += 1
            if self._regime_counts["diversity"] >= self.cfg.regime_confirm:
                action = self._respond_diversity(div, step)
                if action:
                    actions["L3_diversity"] = action
                    self._regime_counts["diversity"] = 0
        else:
            self._regime_counts["diversity"] = 0

        # RadiusDrift (DegEq precursor — CGT insight: rad > rad_max)
        if rad > self.cfg.rad_max:
            self._regime_counts["radius_drift"] += 1
            if self._regime_counts["radius_drift"] >= self.cfg.regime_confirm:
                action = self._respond_radius_drift(rad, step)
                if action:
                    actions["L3_radius_drift"] = action
                    self._regime_counts["radius_drift"] = 0
        else:
            self._regime_counts["radius_drift"] = max(0, self._regime_counts["radius_drift"] - 1)

        # RadiusCollapse risk
        if 0 < rad < self.cfg.rad_min:
            self._regime_counts["radius"] += 1
            if self._regime_counts["radius"] >= self.cfg.regime_confirm:
                action = self._respond_radius(rad, step)
                if action:
                    actions["L3_radius"] = action
                    self._regime_counts["radius"] = 0
        else:
            self._regime_counts["radius"] = 0

        return actions

    def _respond_degeq(self, rdc: float, step: int) -> str:
        """DegEq onset: reduce lambda_distill to dampen radial KL signal."""
        old = self.dcfg.lambda_distill
        new = max(self.cfg.lambda_distill_min, old * 0.7)
        self.dcfg.lambda_distill = new
        if hasattr(self.trainer, 'config'):
            self.trainer.config.lambda_distill = new
        msg = f"rdc={rdc:.2f} > {self.cfg.rdc_danger} — lambda_distill: {old:.4f} → {new:.4f}"
        if self.cfg.verbose:
            print(f"  [L3-DegEq]     ⚠️  {msg}")
        return msg

    def _respond_overconfidence(self, logit_std: float, step: int) -> str:
        """OverconfidenceSpiral: clamp temperature on AngularLMHead."""
        head = getattr(
            getattr(self.trainer.student, "core_model", self.trainer.student),
            "lm_head", None
        )
        if head is not None and hasattr(head, "log_temperature"):
            with torch.no_grad():
                max_log = math.log(8.0)
                head.log_temperature.clamp_(max=max_log)
            msg = f"logit_std={logit_std:.1f} > {self.cfg.logit_std_max} — temp clamped to 8.0"
        else:
            msg = f"logit_std={logit_std:.1f} > {self.cfg.logit_std_max} — no action (no log_temperature)"
        if self.cfg.verbose:
            print(f"  [L3-Overconf]  ⚠️  {msg}")
        return msg

    def _respond_diversity(self, div: float, step: int) -> str:
        """DiversityCollapse: boost temperature slightly to spread distribution."""
        old = self.dcfg.temperature
        new = min(3.0, old * 1.2)
        self.dcfg.temperature = new
        if hasattr(self.trainer, 'config'):
            self.trainer.config.temperature = new
        msg = f"div={div:.2f} < {self.cfg.div_min} — temperature: {old:.2f} → {new:.2f}"
        if self.cfg.verbose:
            print(f"  [L3-Diversity] ⚠️  {msg}")
        return msg

    def _respond_radius(self, rad: float, step: int) -> str:
        """RadiusCollapse risk: boost lambda_radius to pull embeddings back."""
        old = self.dcfg.lambda_radius
        new = min(0.5, old * 1.5)
        self.dcfg.lambda_radius = new
        if hasattr(self.trainer, 'config'):
            self.trainer.config.lambda_radius = new
        msg = f"rad={rad:.4f} < {self.cfg.rad_min} — lambda_radius: {old:.4f} → {new:.4f}"
        if self.cfg.verbose:
            print(f"  [L3-Radius]    ⚠️  {msg}")
        return msg

    def _respond_radius_drift(self, rad: float, step: int) -> str:
        """RadiusDrift: embeddings drifting far from origin — DegEq precursor.
        CGT insight: detect r̄ > rad_max early, reduce lambda_radius to stop drift.
        """
        old = self.dcfg.lambda_radius
        new = min(0.5, old * 1.3)  # boost anchor to pull embeddings back
        self.dcfg.lambda_radius = new
        if hasattr(self.trainer, "config"):
            self.trainer.config.lambda_radius = new
        msg = f"rad={rad:.3f} > {self.cfg.rad_max} (drift) — lambda_radius: {old:.4f} → {new:.4f}"
        if self.cfg.verbose:
            print(f"  [L3-RadiusDrift] ⚠️  {msg}")
        return msg

    # ── PBT (multi-GPU) ──────────────────────────────────────────────────────

    def _pbt_sync(self, step: int) -> Dict[str, Any]:
        """
        PBT synchronisation step.

        Each worker writes its current fitness to a shared file.
        Bottom 20% workers load weights + hyperparameters from top 20%
        and perturb them.

        Requires pbt_shared_dir to be set to a path accessible by all workers
        (e.g. Google Drive, NFS, or torch.distributed shared store).
        """
        if not self.cfg.pbt_shared_dir:
            return {}

        shared = Path(self.cfg.pbt_shared_dir)
        shared.mkdir(parents=True, exist_ok=True)

        # ── Write this worker's fitness ───────────────────────────────────────
        val_ppls = [r.get("val_ppl", 9999) for r in self.trainer.val_hist[-3:]]
        rdcs     = [r.get("rdc_ema", 10)   for r in self.trainer.val_hist[-3:]]
        fitness  = self._compute_pbt_fitness(val_ppls, rdcs)

        worker_state = {
            "worker_id":       self.cfg.worker_id,
            "step":            step,
            "fitness":         fitness,
            "val_ppl":         float(val_ppls[-1]) if val_ppls else 9999,
            "rdc_ema":         float(rdcs[-1])     if rdcs     else 10,
            "lambda_distill":  self.dcfg.lambda_distill,
            "lambda_hidden":   self.dcfg.lambda_hidden,
            "learning_rate":   self.dcfg.learning_rate,
            "temperature":     self.dcfg.temperature,
        }
        (shared / f"worker_{self.cfg.worker_id}.json").write_text(
            json.dumps(worker_state, indent=2)
        )

        # ── Wait for all workers (simple polling) ─────────────────────────────
        import time
        deadline = time.time() + 120   # 2 min timeout
        while time.time() < deadline:
            states = self._read_all_worker_states(shared, step)
            if len(states) == self.cfg.n_workers:
                break
            time.sleep(5)
        else:
            warnings.warn(
                f"[PBT] Timeout waiting for all workers at step {step}. "
                "Skipping exploit/explore this round.",
                UserWarning
            )
            return {}

        # ── Exploit / Explore ─────────────────────────────────────────────────
        states_sorted = sorted(states, key=lambda s: s["fitness"], reverse=True)
        n_exploit      = max(1, int(self.cfg.pbt_exploit_frac * self.cfg.n_workers))
        top_ids        = {s["worker_id"] for s in states_sorted[:n_exploit]}
        bot_ids        = {s["worker_id"] for s in states_sorted[-n_exploit:]}

        actions = {}
        if self.cfg.worker_id in bot_ids:
            # Copy hyperparameters from a random top worker
            import random
            donor = random.choice([s for s in states_sorted if s["worker_id"] in top_ids])
            self._pbt_exploit(donor)
            self._pbt_explore()
            actions["PBT_exploit"] = f"worker_{self.cfg.worker_id} ← worker_{donor['worker_id']}"
            if self.cfg.verbose:
                print(f"  [PBT] Worker {self.cfg.worker_id} exploiting "
                      f"worker {donor['worker_id']} "
                      f"(fitness {fitness:.4f} → {donor['fitness']:.4f})")

        return actions

    def _compute_pbt_fitness(
        self, val_ppls: List[float], rdcs: List[float]
    ) -> float:
        """
        Compute scalar fitness for PBT ranking.

        Higher = better. Combines val_ppl (lower is better),
        rdc_ema (lower is better), and logit_std (target range 1–5).

        Hyperbolic-specific modification vs DeepMind (2017):
        rdc_ema is included as a geometry fidelity term. A model with
        good PPL but DegEq active (rdc*≈10) scores lower than a model
        with slightly worse PPL but rdc*<2.
        """
        w = self.cfg.pbt_fitness_weights
        ppl   = val_ppls[-1] if val_ppls else 9999
        rdc   = rdcs[-1]     if rdcs     else 10.0

        # Normalise: PPL 50–500 → 0–1 (lower = higher score)
        ppl_score = 1.0 - min(1.0, max(0.0, (ppl - 50) / 450))
        # RDC 0–10 → 0–1 (lower = higher score)
        rdc_score = 1.0 - min(1.0, rdc / 10.0)

        return w.get("val_ppl", 0.5) * ppl_score + w.get("rdc_ema", 0.3) * rdc_score

    def _pbt_exploit(self, donor: Dict) -> None:
        """Copy hyperparameters from donor worker."""
        self.dcfg.lambda_distill = donor["lambda_distill"]
        self.dcfg.lambda_hidden  = donor["lambda_hidden"]
        self.dcfg.learning_rate  = donor["learning_rate"]
        self.dcfg.temperature    = donor["temperature"]
        # Update live optimizer LR
        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] = donor["learning_rate"]

    def _pbt_explore(self) -> None:
        """Perturb hyperparameters by ±pbt_perturb_factor."""
        import random
        f = self.cfg.pbt_perturb_factor
        for attr, lo, hi in [
            ("lambda_distill", self.cfg.lambda_distill_min, self.cfg.lambda_distill_max),
            ("lambda_hidden",  self.cfg.lambda_hidden_min,  self.cfg.lambda_hidden_max),
            ("learning_rate",  self.cfg.lr_min,             self.cfg.lr_max),
        ]:
            old = getattr(self.dcfg, attr)
            factor = random.uniform(1.0 - f, 1.0 + f)
            new = float(max(lo, min(hi, old * factor)))
            setattr(self.dcfg, attr, new)
        # Update live optimizer LR
        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] = self.dcfg.learning_rate

    def _read_all_worker_states(self, shared: Path, step: int) -> List[Dict]:
        """Read all worker state files written at the current step."""
        states = []
        for i in range(self.cfg.n_workers):
            p = shared / f"worker_{i}.json"
            if not p.exists():
                continue
            try:
                s = json.loads(p.read_text())
                if s.get("step") == step:
                    states.append(s)
            except Exception:
                pass
        return states

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_grad_balance(self) -> Optional[Dict[str, float]]:
        """Extract latest GradBalance metrics from trainer history."""
        # Look in train_hist for the most recent grad_balance entry
        # The trainer logs GradBalance as a special field
        for m in reversed(getattr(self.trainer, "train_hist", [])):
            if "eff_kl" in m and "eff_hidden" in m:
                return {
                    "eff_kl":  float(m["eff_kl"]),
                    "eff_hid": float(m["eff_hidden"]),
                }
        return None

    def _write_log(self) -> None:
        """Append latest action to controller log CSV."""
        if not self._action_log:
            return
        try:
            log_path = self.cfg.log_path
            if log_path is None:
                # Auto-detect from trainer checkpoint_dir
                ckpt = getattr(self.trainer, "checkpoint_dir", None)
                if ckpt:
                    log_path = str(Path(str(ckpt)).parent.parent /
                                   "logs" / "adaptive_controller.csv")
            if log_path is None:
                return
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            import csv
            exists = Path(log_path).exists()
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(self._action_log[-1].keys()),
                    extrasaction="ignore"
                )
                if not exists:
                    writer.writeheader()
                writer.writerow(self._action_log[-1])
        except Exception:
            pass

    def summary(self) -> str:
        """Return a human-readable summary of controller state."""
        lines = [
            "AdaptiveHyperController summary:",
            f"  lambda_distill : {self.dcfg.lambda_distill:.4f}",
            f"  lambda_hidden  : {self.dcfg.lambda_hidden:.4f}",
            f"  temperature    : {self.dcfg.temperature:.4f}",
            f"  ppl_history    : {[round(p,1) for p in self._ppl_history[-5:]]}",
            f"  stall_count    : {self._stall_count}",
            f"  regime_counts  : {self._regime_counts}",
            f"  actions_taken  : {len(self._action_log)}",
        ]
        return "\n".join(lines)
