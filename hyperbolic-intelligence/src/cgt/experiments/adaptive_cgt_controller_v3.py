# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
adaptive_cgt_controller_v3.py
==============================
V3 of the CGT adaptive controller.

Version history:
  V1: GradBalance by loss magnitude (abs(loss_value))
  V2: + RadiusDrift detection + grad_balance_every=500
  V3: + Real GradNorm (‖∇_θ L_i‖₂) replaces loss magnitude in L1

Key change in V3: Level 1 now uses torch.autograd.grad to compute
actual gradient norms at a probe parameter, instead of using loss
magnitudes as proxy. This fixes the fundamental issue where a small
loss (e.g. KL=0.5) can have massive gradients while a large loss
(e.g. InfoNCE=2.5) has near-zero gradients.

Author: Éric Gustavo Reis de Sena
Date: April 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# GRAD NORM MEASURE (extracted from distillation_v2.py, adapted for CGT)
# ═══════════════════════════════════════════════════════════════════════════

class CGTGradNormMeasure:
    """
    Measures actual per-loss gradient norms via autograd.grad on a probe parameter.
    
    Adapted from HyDRA's GradNormMeasure for the CGT loss structure.
    The probe is the first LayerNorm weight found in the student encoder —
    small enough for fast computation, deep enough to receive gradients
    from all loss terms.
    
    Cost: ~3-6 extra autograd.grad calls (retain_graph=True) every
    measure_every steps. At measure_every=50: ~6% overhead.
    """

    def __init__(self, measure_every: int = 50):
        self.measure_every = measure_every
        self.history: List[Dict] = []
        self._probe_cache: Optional[torch.nn.Parameter] = None

    def _find_probe(self, model: nn.Module) -> Optional[torch.nn.Parameter]:
        """Find a small parameter deep in the encoder as gradient probe."""
        if self._probe_cache is not None:
            return self._probe_cache
        # Priority 1: LayerNorm weight in projector
        for name, p in model.named_parameters():
            if 'norm' in name.lower() and 'weight' in name and p.requires_grad and p.numel() < 512:
                self._probe_cache = p
                return p
        # Priority 2: Last linear layer weight
        for name, p in reversed(list(model.named_parameters())):
            if 'weight' in name and p.requires_grad and p.dim() >= 2:
                self._probe_cache = p
                return p
        # Fallback
        for p in model.parameters():
            if p.requires_grad:
                self._probe_cache = p
                return p
        return None

    def measure(
        self,
        loss_components: Dict[str, torch.Tensor],
        lambda_weights: Dict[str, float],
        model: nn.Module,
        step: int,
    ) -> Dict[str, float]:
        """
        Measure ‖∇_probe (λ_i · L_i)‖₂ for each loss component.
        
        Args:
            loss_components: {name: loss_tensor} — must have valid computation graph
            lambda_weights: {name: current_lambda} — current λ for each loss
            model: student model (to find probe parameter)
            step: current step
            
        Returns:
            {name: grad_norm_float, 'imbalance_ratio': float}
            Empty dict if not a measurement step.
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
            lam = lambda_weights.get(name, 1.0)
            try:
                effective = lam * loss_val
                grad = torch.autograd.grad(
                    effective, probe,
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

        self.history.append({'step': step, **norms})
        return norms


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CGTControllerConfigV3:
    """V3 config — uses real gradient norms instead of loss magnitudes."""

    # ── Level 1: GradNorm Balance ────────────────────────────────────────
    grad_balance_every:  int   = 50     # batches between GradNorm measurements
    balance_tolerance:   float = 3.0    # max allowed gradient norm ratio
    balance_alpha:       float = 0.5    # GradNorm correction strength (0=off, 1=full)
    balance_ema_beta:    float = 0.95   # EMA smoothing for gradient norms
    lambda_floor:        float = 0.001
    lambda_ceil:         float = 5.0
    
    # Loss component names and their λ attribute on the loss module
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
# CONTROLLER V3
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveCGTControllerV3:
    """
    V3: Closed-loop controller with real gradient norm balancing.
    
    Key difference from V1/V2:
    - L1 uses torch.autograd.grad to measure ‖∇_probe(λ_i · L_i)‖₂
    - Balances by actual gradient contribution, not loss magnitude
    - Falls back to loss magnitude if autograd fails
    """

    VERSION = "3.0"

    def __init__(
        self,
        config: CGTControllerConfigV3,
        loss_module: nn.Module,
        optimizer,
        substrate: nn.Module,
        student_model: nn.Module,
    ):
        self.cfg = config
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.substrate = substrate
        self.student_model = student_model

        # L1: GradNorm measure
        self._grad_measure = CGTGradNormMeasure(measure_every=config.grad_balance_every)
        self._grad_ema: Dict[str, float] = {}
        self._batch_count: int = 0
        self._consecutive_balanced: int = 0  # track stability

        # L2: Validation
        self._rho_history: List[float] = []
        self._stall_count: int = 0

        # L3: Geometry
        self._regime_counts: Dict[str, int] = {
            'radius_drift': 0, 'radius_collapse': 0,
            'diversity_collapse': 0, 'f1_spike': 0,
        }

        # Logging
        self._action_log: List[Dict] = []
        self._lambda_history: List[Dict] = []
        self._grad_norm_history: List[Dict] = []

        if config.verbose:
            print(f"  [CGTController V3] Init — GradNorm L1 @ {config.grad_balance_every} batches")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def step_batch(
        self,
        losses: Dict[str, Any],
        student_emb: torch.Tensor,
        epoch: int,
    ) -> Dict[str, Any]:
        """Call after loss.backward(), before optimizer.step()."""
        if not self.cfg.enabled:
            return {}

        self._batch_count += 1
        actions: Dict[str, Any] = {}

        # L1: GradNorm balance (V3 — real gradient norms)
        if self._batch_count % self.cfg.grad_balance_every == 0:
            actions.update(self._level1_gradnorm(losses))

        # L3: Geometry regime
        if self._batch_count % self.cfg.regime_every == 0:
            actions.update(self._level3_geometry(student_emb, epoch))

        if actions and self.cfg.log_history:
            self._action_log.append(
                {'batch': self._batch_count, 'epoch': epoch, **actions})

        return actions

    def step_eval(self, val_rho: float, epoch: int) -> Dict[str, Any]:
        """Call after validation."""
        if not self.cfg.enabled:
            return {}
        return self._level2_velocity(val_rho, epoch)

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 1 V3: REAL GRADNORM BALANCE
    # ══════════════════════════════════════════════════════════════════════

    def _level1_gradnorm(self, losses: Dict[str, Any]) -> Dict[str, Any]:
        """
        V3: Balance using actual ‖∇_probe(λ_i · L_i)‖₂.
        
        Algorithm:
        1. Extract individual loss tensors from the losses dict
        2. Compute gradient norm at probe parameter for each
        3. If imbalance > tolerance, adjust λ weights using GradNorm rule:
           scale_factor = (target_norm / current_norm) ^ alpha
        4. Fall back to loss magnitude if autograd fails
        """
        # Extract loss tensors (must still have computation graph)
        loss_tensors = {}
        for key in self.cfg.loss_keys:
            val = losses.get(key)
            if isinstance(val, torch.Tensor) and val.requires_grad:
                loss_tensors[key] = val
            elif isinstance(val, (int, float)) and val > 0:
                # Scalar — can't measure gradient, will use as fallback
                pass

        # Get current lambda weights
        lambda_weights = {}
        for key, attr in self.cfg.lambda_attrs.items():
            if hasattr(self.loss_module, attr):
                v = getattr(self.loss_module, attr)
                lambda_weights[key] = v.item() if isinstance(v, torch.Tensor) else float(v)

        # Measure real gradient norms
        gnorms = self._grad_measure.measure(
            loss_tensors, lambda_weights, self.student_model, self._batch_count
        )

        if not gnorms or 'imbalance_ratio' not in gnorms:
            # Fallback: use loss magnitude (V1 behavior)
            return self._level1_fallback(losses)

        imbalance = gnorms.pop('imbalance_ratio', 1.0)
        
        # Log gradient norms
        if self.cfg.log_history:
            self._grad_norm_history.append({
                'batch': self._batch_count,
                'imbalance': round(imbalance, 2),
                **{k: round(v, 6) for k, v in gnorms.items() if isinstance(v, float)}
            })

        # Track consecutive balanced steps
        if imbalance <= self.cfg.balance_tolerance:
            self._consecutive_balanced += 1
            if self.cfg.verbose and self._consecutive_balanced % 10 == 0:
                print(f"  [L1-V3] Balanced for {self._consecutive_balanced} consecutive checks "
                      f"(imbalance={imbalance:.1f}x)")
            return {}
        else:
            self._consecutive_balanced = 0

        # Compute target = mean of active gradient norms
        active = {k: v for k, v in gnorms.items() if isinstance(v, float) and v > 1e-12}
        if len(active) < 2:
            return {}

        target_norm = sum(active.values()) / len(active)
        alpha = self.cfg.balance_alpha
        actions: Dict[str, Any] = {}

        for key, current_norm in active.items():
            attr = self.cfg.lambda_attrs.get(key)
            if attr is None or not hasattr(self.loss_module, attr):
                continue

            old_lambda = getattr(self.loss_module, attr)
            if isinstance(old_lambda, torch.Tensor):
                old_lambda = old_lambda.item()

            # GradNorm rule: scale = (target / current) ^ alpha
            scale_factor = (target_norm / current_norm) ** alpha
            # Clamp to prevent violent oscillations
            scale_factor = max(0.5, min(2.0, scale_factor))

            new_lambda = float(max(
                self.cfg.lambda_floor,
                min(self.cfg.lambda_ceil, old_lambda * scale_factor)
            ))

            if abs(new_lambda - old_lambda) / max(old_lambda, 1e-8) > 0.05:
                setattr(self.loss_module, attr, new_lambda)
                direction = '↓' if new_lambda < old_lambda else '↑'
                actions[f'L1_{key}'] = f'{old_lambda:.4f}→{new_lambda:.4f} {direction} (‖∇‖={current_norm:.4f})'
                if self.cfg.verbose:
                    print(f"  [L1-V3] {key}: λ {old_lambda:.4f}→{new_lambda:.4f} {direction} "
                          f"(‖∇‖={current_norm:.4f}, target={target_norm:.4f}, imb={imbalance:.1f}x)")

        # Log lambda snapshot
        if self.cfg.log_history:
            self._lambda_history.append({
                'batch': self._batch_count,
                'imbalance': round(imbalance, 2),
                **self._get_lambdas()
            })

        return actions

    def _level1_fallback(self, losses: Dict[str, Any]) -> Dict[str, Any]:
        """V1 fallback: balance by loss magnitude when autograd fails."""
        active = {}
        for key in self.cfg.loss_keys:
            val = losses.get(key, 0.0)
            if isinstance(val, torch.Tensor):
                val = val.detach().item()
            if abs(val) > 1e-12:
                active[key] = abs(val)

        if len(active) < 2:
            return {}

        vals = list(active.values())
        ratio = max(vals) / min(vals)
        if ratio <= self.cfg.balance_tolerance:
            return {}

        target = sum(vals) / len(vals)
        actions = {}
        for key, mag in active.items():
            attr = self.cfg.lambda_attrs.get(key)
            if attr is None or not hasattr(self.loss_module, attr):
                continue
            old = getattr(self.loss_module, attr)
            if isinstance(old, torch.Tensor): old = old.item()
            correction = target / mag
            dampened = 1.0 + 0.3 * (correction - 1.0)
            new = float(max(self.cfg.lambda_floor, min(self.cfg.lambda_ceil, old * dampened)))
            if abs(new - old) / max(old, 1e-8) > 0.05:
                setattr(self.loss_module, attr, new)
                actions[f'L1_fallback_{key}'] = f'{old:.4f}→{new:.4f}'
        return actions

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 2: VALIDATION VELOCITY (same as V2)
    # ══════════════════════════════════════════════════════════════════════

    def _level2_velocity(self, val_rho: float, epoch: int) -> Dict[str, Any]:
        self._rho_history.append(val_rho)
        if epoch < self.cfg.warmup_epochs or len(self._rho_history) < 2:
            return {}
        window = self._rho_history[-self.cfg.rho_window:]
        if len(window) < 2:
            return {}
        velocity = window[-1] - window[-2]
        actions = {}
        pg_list = (self.optimizer.param_groups
                   if hasattr(self.optimizer, 'param_groups')
                   else self.optimizer.base_optimizer.param_groups)
        if velocity < self.cfg.rho_min_velocity:
            self._stall_count += 1
            if self._stall_count >= self.cfg.rho_stall_patience:
                for pg in pg_list:
                    pg['lr'] = max(self.cfg.lr_min, pg['lr'] * self.cfg.lr_reduce_factor)
                actions['L2'] = f'reduce (Δρ={velocity:.4f})'
                self._stall_count = 0
        else:
            self._stall_count = max(0, self._stall_count - 1)
            if velocity > 3 * self.cfg.rho_min_velocity:
                for pg in pg_list:
                    pg['lr'] = min(self.cfg.lr_max, pg['lr'] * self.cfg.lr_boost_factor)
                actions['L2'] = f'boost (Δρ={velocity:.4f})'
        return actions

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 3: GEOMETRY REGIME (same as V2)
    # ══════════════════════════════════════════════════════════════════════

    def _level3_geometry(self, student_emb: torch.Tensor, epoch: int) -> Dict[str, Any]:
        actions = {}
        with torch.no_grad():
            radii = self.substrate.lorentz_radius(student_emb)
            mean_r = radii.mean().item()
            sp = student_emb[..., 1:]
            sp_n = torch.nn.functional.normalize(sp, dim=-1)
            if sp_n.shape[0] > 1:
                cos = (sp_n @ sp_n.T).fill_diagonal_(0)
                diversity = 1.0 - cos.abs().mean().item()
            else:
                diversity = 1.0
            f1 = self.substrate.manifold_violation(student_emb).item()

        def _check(name, cond, fix_fn, val):
            if cond:
                self._regime_counts[name] += 1
                if self._regime_counts[name] >= self.cfg.regime_confirm:
                    msg = fix_fn(val)
                    self._regime_counts[name] = 0
                    if msg: return {f'L3_{name}': msg}
            else:
                self._regime_counts[name] = max(0, self._regime_counts[name] - 1)
            return {}

        actions.update(_check('radius_drift', mean_r > self.cfg.radius_drift_max,
                              self._fix_radius_drift, mean_r))
        actions.update(_check('radius_collapse', 0 < mean_r < self.cfg.radius_collapse_min,
                              self._fix_radius_collapse, mean_r))
        actions.update(_check('diversity_collapse', diversity < self.cfg.diversity_min,
                              self._fix_diversity, diversity))
        actions.update(_check('f1_spike', f1 > self.cfg.f1_violation_max,
                              self._fix_f1, f1))
        return actions

    def _fix_radius_drift(self, r):
        if not hasattr(self.loss_module, 'radius_weight'): return ''
        old = self._to_float(self.loss_module.radius_weight)
        new = min(1.0, old * self.cfg.radius_lambda_boost)
        self.loss_module.radius_weight = new
        return f'r̄={r:.2f} λ_rad:{old:.4f}→{new:.4f}'

    def _fix_radius_collapse(self, r):
        if not hasattr(self.loss_module, 'radius_weight'): return ''
        old = self._to_float(self.loss_module.radius_weight)
        new = max(1e-5, old * self.cfg.radius_lambda_reduce)
        self.loss_module.radius_weight = new
        return f'r̄={r:.4f} λ_rad:{old:.4f}→{new:.4f}'

    def _fix_diversity(self, d):
        if not hasattr(self.loss_module, 'temp'): return ''
        old = self._to_float(self.loss_module.temp)
        new = min(0.5, old * self.cfg.diversity_temp_boost)
        self.loss_module.temp = new
        return f'div={d:.3f} τ:{old:.3f}→{new:.3f}'

    def _fix_f1(self, v):
        fn = getattr(self.loss_module, 'minkowski_fn', None)
        if fn is None or not hasattr(fn, 'max_weight'): return ''
        old = fn.max_weight
        new = min(2.0, old * 1.5)
        fn.max_weight = new
        return f'F1={v:.2e} w:{old:.2f}→{new:.2f}'

    # ──────────────────────────────────────────────────────────────────────
    # SERIALISATION
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            'version': self.VERSION,
            'grad_ema': self._grad_ema,
            'rho_history': self._rho_history,
            'stall_count': self._stall_count,
            'regime_counts': self._regime_counts,
            'batch_count': self._batch_count,
            'consecutive_balanced': self._consecutive_balanced,
            'lambdas': self._get_lambdas(),
            'action_log': self._action_log[-50:],
            'lambda_history': self._lambda_history[-100:],
            'grad_norm_history': self._grad_norm_history[-100:],
        }

    def load_state_dict(self, d: Dict) -> None:
        self._grad_ema = d.get('grad_ema', {})
        self._rho_history = d.get('rho_history', [])
        self._stall_count = d.get('stall_count', 0)
        self._regime_counts = d.get('regime_counts', self._regime_counts)
        self._batch_count = d.get('batch_count', 0)
        self._consecutive_balanced = d.get('consecutive_balanced', 0)

    def summary(self) -> str:
        lam = self._get_lambdas()
        last_imb = self._grad_norm_history[-1].get('imbalance', '?') if self._grad_norm_history else '?'
        return (
            f"CGTController V{self.VERSION}  batch={self._batch_count}  "
            f"consec_balanced={self._consecutive_balanced}  last_imb={last_imb}\n"
            f"  lambdas:  {self._fmt(lam)}\n"
            f"  ρ_hist:   {[round(r,4) for r in self._rho_history[-5:]]}\n"
            f"  actions:  {len(self._action_log)}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

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
    def grad_norm_history(self) -> List[Dict]:
        return self._grad_norm_history

    @property
    def lambda_history(self) -> List[Dict]:
        return self._lambda_history

    @property
    def action_log(self) -> List[Dict]:
        return self._action_log

    @property
    def action_count(self) -> int:
        return len(self._action_log)

    @property
    def is_stable(self) -> bool:
        """True if imbalance has been < tolerance for 10+ consecutive checks."""
        return self._consecutive_balanced >= 10
