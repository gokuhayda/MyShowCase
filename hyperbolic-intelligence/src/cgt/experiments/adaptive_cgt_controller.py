# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
adaptive_cgt_controller.py
===========================
AdaptiveCGTController — closed-loop hyperparameter control for CGT training.

Three independent control levels at different timescales:

  Level 1 — LossBalance  (every balance_every batches)
      Monitors effective gradient magnitude of each loss term.
      Rebalances λ weights via proportional control so no single
      objective dominates the shared backbone gradient.
      Inspired by GradNorm (Chen et al., ICML 2018).

  Level 2 — Validation velocity  (every validation step)
      Monitors Spearman ρ improvement rate.
      Reduces LR on stall; boosts on fast convergence.

  Level 3 — Geometry regime  (every regime_every batches)
      Detects manifold-specific pathologies:
        - Radius drift:    r̄ > threshold  → boost λ_radius
        - Radius collapse: r̄ < threshold  → reduce λ_radius
        - Diversity collapse: cos_diversity < threshold → boost τ
        - F1 violation spike: Minkowski error rising → boost w_F1

Integration with CGTTrainer
---------------------------
    controller = AdaptiveCGTController(cfg, loss_fn, optimizer, substrate)

    # Inside CGTTrainer.train_epoch():
    for batch in dataloader:
        losses = criterion(batch_s, batch_t, model, epoch)
        losses['total'].backward()
        controller.step_batch(losses, batch_s, epoch)  # ← after backward
        optimizer.step()

    # After validation:
    controller.step_eval(val_rho, epoch)

Author: Éric Gustavo Reis de Sena
Date: April 2026
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

try:
    import numpy as np
except ImportError:
    np = None


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CGTControllerConfig:
    """
    Configuration for AdaptiveCGTController.

    Attributes
    ----------
    balance_every : int
        Batches between Level-1 balance checks.
    balance_tolerance : float
        Max allowed ratio between strongest/weakest loss gradient EMA.
    balance_lr : float
        Correction aggressiveness (0 = off, 1 = full immediate correction).
    balance_ema_beta : float
        EMA smoothing for per-loss gradient magnitudes.
    lambda_floor / lambda_ceil : float
        Hard bounds on any λ weight.
    rho_window : int
        Number of validation evaluations to average for velocity.
    rho_stall_patience : int
        Consecutive low-velocity evals before LR reduction.
    rho_min_velocity : float
        Minimum absolute Spearman ρ improvement per eval.
    regime_every : int
        Batches between Level-3 geometry checks.
    radius_drift_max : float
        Mean geodesic radius above which drift response triggers.
    radius_collapse_min : float
        Mean geodesic radius below which collapse response triggers.
    diversity_min : float
        Cosine diversity (1 - mean|cos|) below which collapse triggers.
    f1_violation_max : float
        Manifold violation above which F1-spike response triggers.
    regime_confirm : int
        Consecutive bad checks before taking action (prevents jitter).
    """

    # ── Level 1: LossBalance ─────────────────────────────────────────────
    balance_every:       int   = 50
    balance_tolerance:   float = 3.0
    balance_lr:          float = 0.3
    balance_ema_beta:    float = 0.95
    lambda_floor:        float = 0.001
    lambda_ceil:         float = 5.0

    loss_keys: List[str] = field(default_factory=lambda: [
        'loss/contrastive', 'loss/distill', 'loss/topo',
        'loss/lipschitz', 'loss/knn',
    ])
    lambda_attrs: Dict[str, str] = field(default_factory=lambda: {
        'loss/contrastive': 'lc',
        'loss/distill':     'ld',
        'loss/topo':        'lt',
        'loss/lipschitz':   'll',
        'loss/knn':         'lknn',
    })

    # ── Level 2: Validation velocity ─────────────────────────────────────
    rho_window:          int   = 5
    rho_stall_patience:  int   = 8
    rho_min_velocity:    float = 0.001
    lr_reduce_factor:    float = 0.7
    lr_boost_factor:     float = 1.05
    lr_min:              float = 1e-6
    lr_max:              float = 1e-3
    warmup_epochs:       int   = 5

    # ── Level 3: Geometry regime ─────────────────────────────────────────
    regime_every:        int   = 20
    radius_drift_max:    float = 8.0
    radius_collapse_min: float = 0.1
    diversity_min:       float = 0.2
    f1_violation_max:    float = 1e-4
    regime_confirm:      int   = 3

    radius_lambda_boost: float = 1.5
    radius_lambda_reduce:float = 0.7
    diversity_temp_boost:float = 1.2

    # ── General ──────────────────────────────────────────────────────────
    enabled:             bool  = True
    verbose:             bool  = True
    log_history:         bool  = True


# ═══════════════════════════════════════════════════════════════════════════
# CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveCGTController:
    """
    Closed-loop controller for CGT multi-objective training.

    Parameters
    ----------
    config : CGTControllerConfig
    loss_module : nn.Module
        The MultiObjectiveLoss (or MultiObjectiveLossHardened) instance.
        Must expose the λ attributes listed in ``config.lambda_attrs``.
    optimizer : torch.optim.Optimizer
        AdamW or RiemannianOptimizerWrapper whose param_groups carry ``lr``.
    substrate : nn.Module
        LorentzSubstrateHardened for geometry diagnostics.
    """

    def __init__(
        self,
        config: CGTControllerConfig,
        loss_module: nn.Module,
        optimizer,
        substrate: nn.Module,
    ):
        self.cfg = config
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.substrate = substrate

        # L1 state
        self._grad_ema: Dict[str, float] = {k: 1.0 for k in config.loss_keys}
        self._batch_count: int = 0

        # L2 state
        self._rho_history: List[float] = []
        self._stall_count: int = 0

        # L3 state
        self._regime_counts: Dict[str, int] = {
            'radius_drift': 0, 'radius_collapse': 0,
            'diversity_collapse': 0, 'f1_spike': 0,
        }

        # Logging
        self._action_log: List[Dict] = []
        self._lambda_history: List[Dict] = []

        if config.verbose:
            print(f"  [CGTController] Init  L1@{config.balance_every}  "
                  f"L2@eval  L3@{config.regime_every}")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def step_batch(
        self,
        losses: Dict[str, Any],
        student_emb: torch.Tensor,
        epoch: int,
    ) -> Dict[str, Any]:
        """Call after backward(), before optimizer.step()."""
        if not self.cfg.enabled:
            return {}

        self._batch_count += 1
        actions: Dict[str, Any] = {}

        self._update_grad_emas(losses)

        if self._batch_count % self.cfg.balance_every == 0:
            actions.update(self._level1_balance())

        if self._batch_count % self.cfg.regime_every == 0:
            actions.update(self._level3_geometry(student_emb, epoch))

        if actions and self.cfg.log_history:
            self._action_log.append(
                {'batch': self._batch_count, 'epoch': epoch, **actions})

        return actions

    def step_eval(self, val_rho: float, epoch: int) -> Dict[str, Any]:
        """Call after each validation with Spearman ρ."""
        if not self.cfg.enabled:
            return {}
        return self._level2_velocity(val_rho, epoch)

    # ──────────────────────────────────────────────────────────────────────
    # SERIALISATION
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            'grad_ema': self._grad_ema,
            'rho_history': self._rho_history,
            'stall_count': self._stall_count,
            'regime_counts': self._regime_counts,
            'batch_count': self._batch_count,
            'lambdas': self._get_lambdas(),
            'action_log': self._action_log,
            'lambda_history': self._lambda_history,
        }

    def load_state_dict(self, d: Dict) -> None:
        self._grad_ema = d.get('grad_ema', self._grad_ema)
        self._rho_history = d.get('rho_history', [])
        self._stall_count = d.get('stall_count', 0)
        self._regime_counts = d.get('regime_counts', self._regime_counts)
        self._batch_count = d.get('batch_count', 0)
        self._action_log = d.get('action_log', [])
        self._lambda_history = d.get('lambda_history', [])

    def summary(self) -> str:
        lam = self._get_lambdas()
        return (
            f"CGTController  batch={self._batch_count}  "
            f"stall={self._stall_count}  "
            f"regimes={self._regime_counts}\n"
            f"  grad_ema: {self._fmt(self._grad_ema)}\n"
            f"  lambdas:  {self._fmt(lam)}\n"
            f"  ρ_hist:   {[round(r,4) for r in self._rho_history[-5:]]}\n"
            f"  actions:  {len(self._action_log)}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 1 — LOSS BALANCE
    # ══════════════════════════════════════════════════════════════════════

    def _update_grad_emas(self, losses: Dict[str, Any]) -> None:
        β = self.cfg.balance_ema_beta
        for k in self.cfg.loss_keys:
            v = losses.get(k, 0.0)
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
            self._grad_ema[k] = β * self._grad_ema[k] + (1 - β) * abs(v)

    def _level1_balance(self) -> Dict[str, Any]:
        """
        Proportional λ-rebalancing.

        Computes the ratio of the strongest to weakest loss-magnitude EMA.
        If the ratio exceeds ``balance_tolerance``, each λ is nudged toward
        the mean magnitude by a factor of ``balance_lr``.
        """
        active = {k: max(v, 1e-12)
                  for k, v in self._grad_ema.items() if v > 1e-12}
        if len(active) < 2:
            return {}

        vals = list(active.values())
        ratio = max(vals) / min(vals)
        if ratio <= self.cfg.balance_tolerance:
            return {}

        target = sum(vals) / len(vals)
        actions: Dict[str, Any] = {}

        for key, ema in active.items():
            attr = self.cfg.lambda_attrs.get(key)
            if attr is None or not hasattr(self.loss_module, attr):
                continue

            old = getattr(self.loss_module, attr)
            if isinstance(old, torch.Tensor):
                old = old.item()

            correction = target / ema
            dampened = 1.0 + self.cfg.balance_lr * (correction - 1.0)
            new = float(max(self.cfg.lambda_floor,
                            min(self.cfg.lambda_ceil, old * dampened)))

            if abs(new - old) / max(old, 1e-8) > 0.05:
                setattr(self.loss_module, attr, new)
                actions[f'L1_{key}'] = f'{old:.4f}→{new:.4f}'
                if self.cfg.verbose:
                    arrow = '↓' if new < old else '↑'
                    print(f"  [L1] {key}: λ {old:.4f}→{new:.4f} {arrow}"
                          f"  (ema={ema:.4f}, ratio={ratio:.1f})")

        if self.cfg.log_history:
            self._lambda_history.append(
                {'batch': self._batch_count, **self._get_lambdas()})

        return actions

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 2 — VALIDATION VELOCITY
    # ══════════════════════════════════════════════════════════════════════

    def _level2_velocity(self, val_rho: float, epoch: int) -> Dict[str, Any]:
        """
        Adjust LR based on Spearman ρ improvement velocity.

        velocity = ρ_current − ρ_previous  (absolute)
        """
        self._rho_history.append(val_rho)

        if epoch < self.cfg.warmup_epochs or len(self._rho_history) < 2:
            return {}

        window = self._rho_history[-self.cfg.rho_window:]
        if len(window) < 2:
            return {}

        velocity = window[-1] - window[-2]
        actions: Dict[str, Any] = {}

        # Access param_groups (works for both Optimizer and Wrapper)
        pg_list = (self.optimizer.param_groups
                   if hasattr(self.optimizer, 'param_groups')
                   else self.optimizer.base_optimizer.param_groups)

        if velocity < self.cfg.rho_min_velocity:
            self._stall_count += 1
            if self._stall_count >= self.cfg.rho_stall_patience:
                for pg in pg_list:
                    old = pg['lr']
                    pg['lr'] = max(self.cfg.lr_min,
                                   old * self.cfg.lr_reduce_factor)
                actions['L2_lr'] = f'reduce (Δρ={velocity:.4f})'
                self._stall_count = 0
                if self.cfg.verbose:
                    print(f"  [L2] ρ stall Δ={velocity:.4f}  "
                          f"LR ×{self.cfg.lr_reduce_factor}")
        else:
            self._stall_count = max(0, self._stall_count - 1)
            if velocity > 3 * self.cfg.rho_min_velocity:
                for pg in pg_list:
                    old = pg['lr']
                    pg['lr'] = min(self.cfg.lr_max,
                                   old * self.cfg.lr_boost_factor)
                actions['L2_lr'] = f'boost (Δρ={velocity:.4f})'

        return actions

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 3 — GEOMETRY REGIME
    # ══════════════════════════════════════════════════════════════════════

    def _level3_geometry(
        self,
        student_emb: torch.Tensor,
        epoch: int,
    ) -> Dict[str, Any]:
        """Detect and respond to geometric pathologies."""
        actions: Dict[str, Any] = {}

        with torch.no_grad():
            # ── Radius ────────────────────────────────────────────────────
            radii = self.substrate.lorentz_radius(student_emb)
            mean_r = radii.mean().item()

            # ── Diversity (1 − mean |cosine|) ─────────────────────────────
            sp = student_emb[..., 1:]
            sp_n = torch.nn.functional.normalize(sp, dim=-1)
            if sp_n.shape[0] > 1:
                cos = (sp_n @ sp_n.T).fill_diagonal_(0)
                diversity = 1.0 - cos.abs().mean().item()
            else:
                diversity = 1.0

            # ── F1 violation ──────────────────────────────────────────────
            f1 = self.substrate.manifold_violation(student_emb).item()

        # Radius drift
        if mean_r > self.cfg.radius_drift_max:
            actions.update(
                self._regime_check('radius_drift',
                                   self._fix_radius_drift, mean_r))
        else:
            self._regime_counts['radius_drift'] = max(
                0, self._regime_counts['radius_drift'] - 1)

        # Radius collapse
        if 0 < mean_r < self.cfg.radius_collapse_min:
            actions.update(
                self._regime_check('radius_collapse',
                                   self._fix_radius_collapse, mean_r))
        else:
            self._regime_counts['radius_collapse'] = 0

        # Diversity collapse
        if diversity < self.cfg.diversity_min:
            actions.update(
                self._regime_check('diversity_collapse',
                                   self._fix_diversity, diversity))
        else:
            self._regime_counts['diversity_collapse'] = 0

        # F1 spike
        if f1 > self.cfg.f1_violation_max:
            actions.update(
                self._regime_check('f1_spike',
                                   self._fix_f1, f1))
        else:
            self._regime_counts['f1_spike'] = 0

        return actions

    def _regime_check(self, name: str, fix_fn, value: float) -> Dict:
        self._regime_counts[name] += 1
        if self._regime_counts[name] >= self.cfg.regime_confirm:
            msg = fix_fn(value)
            self._regime_counts[name] = 0
            if msg:
                return {f'L3_{name}': msg}
        return {}

    # ── Regime responses ──────────────────────────────────────────────────

    def _fix_radius_drift(self, mean_r: float) -> str:
        if not hasattr(self.loss_module, 'radius_weight'):
            return ''
        old = self._to_float(self.loss_module.radius_weight)
        new = min(1.0, old * self.cfg.radius_lambda_boost)
        self.loss_module.radius_weight = new
        if self.cfg.verbose:
            print(f"  [L3] Radius drift r̄={mean_r:.2f}  "
                  f"λ_rad {old:.4f}→{new:.4f}")
        return f'r̄={mean_r:.2f} λ_rad:{old:.4f}→{new:.4f}'

    def _fix_radius_collapse(self, mean_r: float) -> str:
        if not hasattr(self.loss_module, 'radius_weight'):
            return ''
        old = self._to_float(self.loss_module.radius_weight)
        new = max(1e-5, old * self.cfg.radius_lambda_reduce)
        self.loss_module.radius_weight = new
        if self.cfg.verbose:
            print(f"  [L3] Radius collapse r̄={mean_r:.4f}  "
                  f"λ_rad {old:.4f}→{new:.4f}")
        return f'r̄={mean_r:.4f} λ_rad:{old:.4f}→{new:.4f}'

    def _fix_diversity(self, div: float) -> str:
        if not hasattr(self.loss_module, 'temp'):
            return ''
        old = self._to_float(self.loss_module.temp)
        new = min(0.5, old * self.cfg.diversity_temp_boost)
        self.loss_module.temp = new
        if self.cfg.verbose:
            print(f"  [L3] Diversity collapse div={div:.3f}  "
                  f"τ {old:.3f}→{new:.3f}")
        return f'div={div:.3f} τ:{old:.3f}→{new:.3f}'

    def _fix_f1(self, f1: float) -> str:
        fn = getattr(self.loss_module, 'minkowski_fn', None)
        if fn is None or not hasattr(fn, 'max_weight'):
            return ''
        old = self._to_float(fn.max_weight)
        new = min(2.0, old * 1.5)
        fn.max_weight = new
        if self.cfg.verbose:
            print(f"  [L3] F1 spike={f1:.2e}  w_F1 {old:.2f}→{new:.2f}")
        return f'F1={f1:.2e} w:{old:.2f}→{new:.2f}'

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_lambdas(self) -> Dict[str, float]:
        out = {}
        for _, attr in self.cfg.lambda_attrs.items():
            if hasattr(self.loss_module, attr):
                out[attr] = self._to_float(getattr(self.loss_module, attr))
        return out

    @staticmethod
    def _to_float(v) -> float:
        return v.item() if isinstance(v, torch.Tensor) else float(v)

    @staticmethod
    def _fmt(d: Dict, p: int = 4) -> str:
        return '{' + ', '.join(
            f'{k}={v:.{p}f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in d.items()) + '}'

    @property
    def lambda_history(self) -> List[Dict]:
        return self._lambda_history

    @property
    def action_log(self) -> List[Dict]:
        return self._action_log
