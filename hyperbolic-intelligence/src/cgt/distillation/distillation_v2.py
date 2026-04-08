"""
cgt/distillation/distillation_v2.py
=========================================
Isolated distillation stack for v2.

Provides:
    DistillationConfigV2        — config with precedence rules
    GPT2TeacherWrapperV2        — frozen GPT-2 teacher
    TeacherDistillationLossV2   — KL + CE + hidden alignment
    DistillationTrainerV2       — full training loop

All losses use v2 geometry (LorentzSubstrateV2).
Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cgt.geometry import LorentzSubstrateV2


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping V3  —  Dual-EMA Reference Tracking
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStoppingV3:
    """
    Dual-reference EMA early stopping for noise-limited LM training regimes.

    Problem with single-reference EMA stopping
    -------------------------------------------
    Standard EMA-based stopping tracks one reference ``best_ema`` and fires
    ``is_best`` when the EMA drops by more than ``min_delta`` relative to that
    reference.  After long training with ``beta = 0.9``, the EMA can lag raw
    validation loss by 0.05–0.10 units.  On a true plateau, the EMA slowly
    converges toward the real value, crossing the ``min_delta = 0.5 %``
    threshold at every step.  Result: ``is_best`` fires perpetually and
    patience never accumulates.

    Dual-reference solution
    -----------------------
    Two EMA tracks with different time constants solve the conflict:

    Slow EMA  (``ema_beta`` = 0.9)
        Long memory.  Used for *phase detection* (slope of the trend) and the
        ``smoothed_val`` display metric.  Never used for stopping decisions.

    Fast EMA  (``ema_beta_fast`` = 0.3)
        Short memory (~4 evals to 95 % convergence).  Responds within a few
        evaluations to genuine improvement or to a plateau.  Used for all
        stopping decisions.

    Two reference points on the fast EMA
    -------------------------------------
    ``best_ema_global``
        Strict running minimum of the fast EMA.  Updated only when the fast
        EMA improves by more than ``min_delta`` (0.5 %).
        Never decays; represents the best performance the model has achieved.

    ``best_ema_local``  (implicit — computed dynamically)
        Maximum of the fast EMA over the last ``window_size`` evaluations.
        This is the *sliding window peak* — the highest recent fast-EMA value.
        Improvement = ``best_ema_local − current_ema_fast`` measures how much
        the fast EMA has descended within the recent window.
        As the model plateaus, the window fills with nearly-equal values, so
        ``best_ema_local ≈ current_ema_fast`` and improvement ≈ 0.
        No explicit re-anchoring is needed — the sliding window naturally
        tracks the current operating regime.

    Noise-aware patience gate
    -------------------------
    Given ``local_improvement = max(fast_ema_window) − current_ema_fast``:

    -  ``local_improvement > noise_mult × noise_level``
       The fast EMA is descending faster than the noise floor → genuine local
       improvement → **hold** patience.

    -  ``local_improvement ≤ noise_mult × noise_level``
       Movement is within the noise envelope → noise-limited regime →
       **increment** patience.

    Noise is estimated as the standard deviation of the *detrended* raw
    validation window.  Detrending removes the contribution of genuine descent
    from the variance estimate, isolating stochastic scatter.

    Dual stopping gate
    ------------------
    A stop fires only when ALL conditions hold:

    1. ``patience ≥ effective_patience``  (local patience exhausted)
    2. ``steps_since_global ≥ plateau_threshold``  (global stagnation confirmed)
    3. logit_std growth ≤ ``logit_std_delta`` over window (model specialisation done)
    4. ``current_step ≥ min_steps``  (warmup guard)

    Condition 2 prevents stopping when a lucky noise sample briefly fires
    ``is_best_global`` and resets patience, even if the model is genuinely
    on a plateau.  The model must have been globally stagnant for at least
    ``plateau_threshold`` consecutive evaluations.

    Phase detection
    ---------------
    OLS slope of the fast-EMA window determines the current phase:

    - Phase 1  ``|slope| > 0.05``  — rapid descent
    - Phase 2  ``|slope| > 0.005`` — gradual convergence
    - Phase 3  ``|slope| ≤ 0.005`` — noise-limited refinement

    In Phase 3 the effective patience is scaled by
    ``phase3_patience_multiplier``.

    Contribution
    ------------
    The dual-reference design isolates two distinct failure modes of
    EMA-based stopping:

    *  **Global stagnation** (``best_ema_global``): detected by the absence
       of >0.5 % improvement in the fast EMA over ``plateau_threshold`` steps.

    *  **Local noise floor** (sliding window max): detected by the fast EMA
       failing to descend beyond the detrended noise level within the window.

    Stopping requires *both* to agree — neither alone is sufficient in the
    presence of EMA lag or stochastic plateau noise.

    API
    ---
    >>> stopper = EarlyStoppingV3()
    >>> should_stop, info = stopper.step(val_loss, logit_std=logit_std)
    >>> if should_stop:
    ...     print(info["decision"])

    Parameters
    ----------
    patience : int
        Base patience limit.  Default 10.
    ema_beta : float
        Slow EMA decay (phase detection + display).  Default 0.9.
    ema_beta_fast : float
        Fast EMA decay (stopping decisions).  Default 0.3.
    min_delta : float
        Relative improvement threshold for ``is_best_global``.
        ``(best_ema_global − fast_ema) / best_ema_global > min_delta``.
        Default 0.005 (0.5 %).
    window_size : int
        Window for noise estimation, phase slope, and sliding local reference.
        Default 5.
    noise_mult : float
        Multiplier on noise level for the "hold" gate.  Default 2.0.
    min_steps : int
        Warmup guard: no stop before this step.  Default 1500.
    plateau_threshold : int
        Minimum consecutive evaluations without global improvement before a
        stop can fire.  Also the re-anchor cadence in earlier implementations.
        Default 5.
    logit_std_delta : float
        logit_std growth gate threshold across window.  Default 0.1.
    phase3_patience_multiplier : float
        Patience scale factor in Phase 3.  Default 1.0.
    eval_every : int
        Evaluations per training step (for step estimation when
        ``current_step`` is None).  Default 200.
    """

    def __init__(
        self,
        patience:                   int   = 10,
        ema_beta:                   float = 0.9,
        ema_beta_fast:              float = 0.3,
        min_delta:                  float = 0.005,
        window_size:                int   = 5,
        noise_mult:                 float = 2.0,
        min_steps:                  int   = 1500,
        plateau_threshold:          int   = 5,
        logit_std_delta:            float = 0.1,
        phase3_patience_multiplier: float = 1.0,
        eval_every:                 int   = 200,
    ) -> None:
        self.patience                   = patience
        self.ema_beta                   = ema_beta
        self.ema_beta_fast              = ema_beta_fast
        self.min_delta                  = min_delta
        self.window_size                = window_size
        self.noise_mult                 = noise_mult
        self.min_steps                  = min_steps
        self.plateau_threshold          = plateau_threshold
        self.logit_std_delta            = logit_std_delta
        self.phase3_patience_multiplier = phase3_patience_multiplier
        self.eval_every                 = eval_every

        # ── mutable state (serialised in checkpoints) ─────────────────────
        # Slow EMA (display + phase)
        self._ema_slow_raw: float           = 0.0
        self._ema_slow:     Optional[float] = None

        # Fast EMA (stopping decisions)
        self._ema_fast_raw: float           = 0.0
        self._ema_fast:     Optional[float] = None

        # Dual references
        self._best_ema_global: float        = float("inf")  # strict running min
        self._best_raw:        float        = float("inf")  # raw val at last global best

        # Histories
        self._raw_window:  List[float]      = []   # for detrended noise
        self._fast_window: List[float]      = []   # for sliding local ref + phase
        self._std_window:  List[float]      = []   # for logit_std gate

        # Counters
        self._patience_count:    int        = 0
        self._steps_since_global: int       = 0
        self._n_calls:           int        = 0

    # ── public API ────────────────────────────────────────────────────────

    def step(
        self,
        val_loss:     float,
        logit_std:    float = 0.0,
        current_step: Optional[int] = None,
    ) -> tuple:
        """
        Process one validation measurement.

        Parameters
        ----------
        val_loss : float
            Raw validation loss.
        logit_std : float, optional
            Standard deviation of model logits.
        current_step : int, optional
            Actual training step.  Estimated from call count × ``eval_every``
            when None.

        Returns
        -------
        (should_stop, debug_info) : (bool, dict)
            debug_info keys: ema, ema_fast, noise, phase, rel_improvement,
            patience, effective_patience, is_best, local_improvement,
            steps_since_global, decision.
        """
        self._n_calls += 1
        effective_step = (
            current_step if current_step is not None
            else self._n_calls * self.eval_every
        )

        # ── Slow EMA (bias-corrected) — display + phase ───────────────────
        bc_slow = 1.0 - self.ema_beta ** self._n_calls
        self._ema_slow_raw = (
            self.ema_beta * self._ema_slow_raw
            + (1.0 - self.ema_beta) * val_loss
        )
        ema_slow = self._ema_slow_raw / max(bc_slow, 1e-8)
        self._ema_slow = ema_slow

        # ── Fast EMA (bias-corrected) — stopping decisions ────────────────
        bc_fast = 1.0 - self.ema_beta_fast ** self._n_calls
        self._ema_fast_raw = (
            self.ema_beta_fast * self._ema_fast_raw
            + (1.0 - self.ema_beta_fast) * val_loss
        )
        ema_fast = self._ema_fast_raw / max(bc_fast, 1e-8)
        self._ema_fast = ema_fast

        # ── Update windows ────────────────────────────────────────────────
        self._raw_window.append(val_loss)
        if len(self._raw_window) > self.window_size:
            self._raw_window = self._raw_window[-self.window_size:]

        self._fast_window.append(ema_fast)
        if len(self._fast_window) > self.window_size:
            self._fast_window = self._fast_window[-self.window_size:]

        self._std_window.append(logit_std)
        if len(self._std_window) > self.window_size:
            self._std_window = self._std_window[-self.window_size:]

        # ── Phase detection (fast EMA slope) ─────────────────────────────
        fast_slope = self._ols_slope(self._fast_window)
        if fast_slope < -0.05:
            phase = 1
        elif fast_slope < -0.005:
            phase = 2
        else:
            phase = 3

        effective_patience = (
            int(self.patience * self.phase3_patience_multiplier)
            if phase == 3 else self.patience
        )

        # ── Detrended noise (raw window) ──────────────────────────────────
        noise_level = self._detrended_std(self._raw_window)

        # ── Global is_best (fast EMA vs strict running minimum) ───────────
        if math.isinf(self._best_ema_global):
            is_best_global = True
        else:
            rel_improvement = (
                (self._best_ema_global - ema_fast)
                / max(abs(self._best_ema_global), 1e-8)
            )
            is_best_global = rel_improvement > self.min_delta

        if is_best_global:
            self._best_ema_global  = ema_fast
            self._best_raw         = val_loss
            self._patience_count   = 0
            self._steps_since_global = 0
            noise_gate             = "is_best"
        else:
            self._steps_since_global += 1

            # ── Local reference: sliding window maximum of fast EMA ───────
            # As the fast EMA plateaus, max(window) ≈ current fast EMA, so
            # local_improvement ≈ 0 and patience accumulates.
            # During genuine descent, max(window) = oldest (highest) value,
            # so local_improvement > 0 and patience is held.
            # No explicit re-anchoring is required — the window slides naturally.
            best_ema_local     = max(self._fast_window)
            local_improvement  = best_ema_local - ema_fast

            if local_improvement > self.noise_mult * noise_level:
                noise_gate = "hold"            # descending above noise floor
            else:
                self._patience_count += 1      # noise-limited plateau
                noise_gate = "pat++"

        # ── Assemble debug dict ───────────────────────────────────────────
        rel_impr = (
            (self._best_ema_global - ema_fast)
            / max(abs(self._best_ema_global), 1e-8)
            if not math.isinf(self._best_ema_global) else 1.0
        )
        local_impr = (max(self._fast_window) - ema_fast) if self._fast_window else 0.0

        debug: Dict[str, Any] = {
            "ema":                ema_slow,        # slow EMA (display)
            "ema_fast":           ema_fast,        # fast EMA (decisions)
            "smoothed_val":       ema_slow,        # alias for trainer compat
            "rel_improvement":    rel_impr,
            "noise":              noise_level,
            "phase":              phase,
            "patience":           self._patience_count,
            "patience_count":     self._patience_count,
            "effective_patience": effective_patience,
            "is_best":            is_best_global,
            "local_improvement":  local_impr,
            "steps_since_global": self._steps_since_global,
            "noise_gate":         noise_gate,
            "decision":           None,
            "should_stop":        False,
        }

        # ── Gate 1: warmup guard ──────────────────────────────────────────
        if effective_step < self.min_steps:
            debug["decision"] = (
                f"warmup-guard (step {effective_step} < {self.min_steps})"
            )
            return False, debug

        # ── Gate 2: patience not exhausted ───────────────────────────────
        if self._patience_count < effective_patience:
            debug["decision"] = ""
            return False, debug

        # ── Gate 3: global stagnation not confirmed ───────────────────────
        # Requires at least plateau_threshold evals without global improvement.
        # Without this gate, a lucky noise sample that fires is_best_global
        # (resetting steps_since_global to 0) would prevent stopping even on
        # a true plateau where the model is not making real progress.
        if self._steps_since_global < self.plateau_threshold:
            debug["decision"] = (
                f"global-not-stagnant (sg={self._steps_since_global}"
                f" < {self.plateau_threshold})"
            )
            return False, debug

        # ── Gate 4: logit_std still growing ──────────────────────────────
        if len(self._std_window) >= 2:
            std_growth = self._std_window[-1] - self._std_window[0]
            if std_growth > self.logit_std_delta:
                self._patience_count = max(0, self._patience_count - 1)
                debug["patience"]       = self._patience_count
                debug["patience_count"] = self._patience_count
                debug["decision"] = (
                    f"logit_std-growing Δ={std_growth:.4f}"
                    f" > {self.logit_std_delta}"
                )
                return False, debug

        # ── All gates passed → STOP ───────────────────────────────────────
        stop_reason = (
            f"patience={self._patience_count}/{effective_patience}  |  "
            f"sg={self._steps_since_global}/{self.plateau_threshold}  |  "
            f"ema_fast={ema_fast:.4f}  best_global={self._best_ema_global:.4f}  "
            f"rel_Δ={rel_impr:.5f}  |  "
            f"noise={noise_level:.5f}  local_Δ={local_impr:.5f}  |  "
            f"phase=P{phase}"
        )
        debug["decision"]    = stop_reason
        debug["should_stop"] = True
        return True, debug

    # ── serialisation ─────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Return all mutable state for checkpoint persistence."""
        return {
            "ema_slow_raw":       self._ema_slow_raw,
            "ema_slow":           self._ema_slow,
            "ema_fast_raw":       self._ema_fast_raw,
            "ema_fast":           self._ema_fast,
            "best_ema_global":    self._best_ema_global,
            "best_raw":           self._best_raw,
            "raw_window":         list(self._raw_window),
            "fast_window":        list(self._fast_window),
            "std_window":         list(self._std_window),
            "patience_count":     self._patience_count,
            "steps_since_global": self._steps_since_global,
            "n_calls":            self._n_calls,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Restore from checkpoint.  Forward-compatible with V3 checkpoints."""
        self._ema_slow_raw       = d.get("ema_slow_raw",       0.0)
        self._ema_slow           = d.get("ema_slow",           None)
        self._ema_fast_raw       = d.get("ema_fast_raw",       0.0)
        self._ema_fast           = d.get("ema_fast",           None)
        self._best_ema_global    = d.get("best_ema_global",    float("inf"))
        self._best_raw           = d.get("best_raw",           float("inf"))
        self._raw_window         = list(d.get("raw_window",    []))
        self._fast_window        = list(d.get("fast_window",   []))
        self._std_window         = list(d.get("std_window",    []))
        self._patience_count     = d.get("patience_count",     0)
        self._steps_since_global = d.get("steps_since_global", 0)
        self._n_calls            = d.get("n_calls",            0)

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _ols_slope(values: List[float]) -> float:
        """Numerically stable OLS slope for a short list."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2                   for i in range(n))
        return num / max(den, 1e-10)

    @staticmethod
    def _detrended_std(values: List[float]) -> float:
        """
        Std of residuals after removing a linear trend.

        Detrending prevents genuine descent from inflating the noise estimate:
        raw std of [6.3, 6.2, 6.1, 6.0, 5.9] is 0.16, while the actual
        stochastic scatter is near zero.  Residuals from the OLS fit give
        the true measurement noise.
        """
        n = len(values)
        if n < 2:
            return 0.0
        slope     = EarlyStoppingV3._ols_slope(values)
        detrended = [values[i] - slope * i for i in range(n)]
        mean      = sum(detrended) / n
        return math.sqrt(
            sum((v - mean) ** 2 for v in detrended) / max(n - 1, 1)
        )

# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class HiddenProjector(nn.Module):
    """
    Robust early stopping for noisy hyperbolic LM training curves.

    Architecture — dual-track signal processing
    --------------------------------------------
    A single EMA causes a fundamental conflict: a high ``beta`` (e.g. 0.9)
    smooths noise well but creates catastrophic lag on the improvement check —
    the EMA keeps converging asymptotically toward a plateau from above,
    registering a spurious "new best" at every step and preventing the
    patience counter from ever firing.

    This class solves the conflict by separating the two concerns:

    Track A — Improvement / patience  (raw ``current_val``)
        Patience is counted against the **raw** validation loss, compared to
        ``best_raw`` with a *relative* threshold.  Raw values respond within
        one eval step to a true plateau.  Single-step noise is tolerated by
        the relative threshold (0.5 % by default) — a one-step blip of < 0.5 %
        will not reset patience, but sustained improvement will.

    Track B — Trend gate  (OLS slope over last N raw values)
        The Ordinary Least Squares slope is itself a smoothed estimator
        (it minimises the sum of squared residuals over the window).
        No extra EMA is needed for the trend check.  If the slope is still
        significantly negative when patience is exhausted, the patience
        counter is decremented to give the model more time.

    Track C — logit_std multi-signal gate
        If ``logit_std`` is still growing across the window the model is
        still specialising its output distribution.  Patience is slowed.

    Heavy EMA (``ema_beta`` = 0.9) is maintained for the ``smoothed_val``
    log metric ONLY — it is never used in any decision path.

    Four-layer decision process
    ---------------------------
    1. Warmup guard    — Never stop before ``min_steps_before_stopping``.
    2. Raw δ patience  — (best_raw − current) / best_raw > min_delta resets.
    3. Trend gate      — OLS slope over last N raw values; still descending
                         → decrement patience counter by 1 (slow the clock).
    4. logit_std gate  — logit_std still growing → decrement patience by 1.

    All state is serialisable via ``state_dict`` / ``load_state_dict`` so
    checkpoints survive runtime restarts without resetting counters.

    Parameters
    ----------
    patience : int
        Consecutive eval steps with no relative improvement before
        ``should_stop`` can be set.  Default 8.
    ema_beta : float
        Decay for the display-only ``smoothed_val`` metric.  Has no effect
        on any stopping decision.  Default 0.9.
    min_delta : float
        Relative improvement threshold for the patience check.
        ``(best_raw − current_val) / best_raw > min_delta`` resets patience.
        0.005 = 0.5 %.  Default 0.005.
    min_steps_before_stopping : int
        Hard lower bound on training steps.  Default 1500.
    trend_window : int
        Number of most-recent raw values used for slope estimation and
        logit_std growth check.  Default 5.
    logit_std_min_delta : float
        Minimum logit_std growth across the window to keep the gate open.
        Default 0.05.

    Usage
    -----
    >>> stopper = EarlyStoppingV3(patience=10, min_steps=1500)
    >>> result  = stopper.step(val_loss, current_step, logit_std=logit_std)
    >>> if result["should_stop"]:
    ...     print("Stop:", result["reason"])
    ...     break
    """

    def __init__(
        self,
        patience:                   int   = 8,
        ema_beta:                   float = 0.9,
        min_delta:                  float = 0.005,
        min_steps_before_stopping:  int   = 1500,
        trend_window:               int   = 5,
        logit_std_min_delta:        float = 0.05,
    ) -> None:
        self.patience              = patience
        self.ema_beta              = ema_beta
        self.min_delta             = min_delta
        self.min_steps             = min_steps_before_stopping
        self.trend_window          = trend_window
        self.logit_std_min_delta   = logit_std_min_delta

        # ── mutable state (persisted in checkpoints) ──────────────────────
        # Track A: raw improvement
        self.best_raw:           float           = float("inf")
        self.patience_count:     int             = 0
        # Track B: trend gate (raw values)
        self.val_history:        List[float]     = []   # last trend_window raw vals
        # Track C: logit_std gate
        self.logit_std_history:  List[float]     = []   # last trend_window stds
        # Display-only heavy EMA
        self.ema_val:            Optional[float] = None
        self._n_evals:           int             = 0

    # ── public API ────────────────────────────────────────────────────────

    def step(
        self,
        current_val:   float,
        current_step:  int,
        logit_std:     float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process one validation measurement.

        Returns
        -------
        dict
            smoothed_val   : display-only heavy-EMA value (not used in decisions)
            is_best        : True when current_val is a new relative best
            patience_count : current patience counter
            should_stop    : True when all gates clear and patience is exhausted
            reason         : human-readable explanation; empty string when running
        """
        self._n_evals += 1

        # ── Display-only EMA (no decision role) ───────────────────────────
        if self.ema_val is None:
            self.ema_val = current_val
        else:
            self.ema_val = self.ema_beta * self.ema_val + (1.0 - self.ema_beta) * current_val
        smoothed = self.ema_val

        # ── Track B: raw val history for trend gate ───────────────────────
        self.val_history.append(current_val)
        if len(self.val_history) > self.trend_window:
            self.val_history = self.val_history[-self.trend_window:]

        # ── Track C: logit_std history ────────────────────────────────────
        self.logit_std_history.append(logit_std)
        if len(self.logit_std_history) > self.trend_window:
            self.logit_std_history = self.logit_std_history[-self.trend_window:]

        # ── Track A: relative improvement against best_raw ────────────────
        # First call is always a best (initialisation from inf is well-defined).
        if math.isinf(self.best_raw):
            is_best = True
        else:
            rel_improvement = (self.best_raw - current_val) / max(abs(self.best_raw), 1e-8)
            is_best = rel_improvement > self.min_delta

        if is_best:
            self.best_raw       = current_val
            self.patience_count = 0
        else:
            self.patience_count += 1

        # ── Gate 1: Warmup guard ──────────────────────────────────────────
        if current_step < self.min_steps:
            return self._result(
                smoothed, is_best,
                should_stop=False,
                reason=f"[warmup-guard] step {current_step} < min_steps {self.min_steps}",
            )

        # ── Gate 2: Patience not yet exhausted ────────────────────────────
        if self.patience_count < self.patience:
            return self._result(smoothed, is_best, should_stop=False, reason="")

        # ── Patience exhausted — run secondary checks before stopping ─────

        # ── Gate 3: Trend gate (OLS slope on raw history) ─────────────────
        # OLS over N points is inherently a smoothed estimator; no extra EMA needed.
        # A still-negative slope means the model is converging slowly, not stalled.
        if len(self.val_history) >= self.trend_window:
            slope = self._linear_slope(self.val_history)
            if slope < -1e-4:
                # Slow the stop clock — one decrement buys one extra eval.
                self.patience_count = max(0, self.patience_count - 1)
                return self._result(
                    smoothed, is_best,
                    should_stop=False,
                    reason=f"[trend-active] OLS slope={slope:.5f}",
                )

        # ── Gate 4: logit_std multi-signal ───────────────────────────────
        # Growing logit_std → model is still differentiating output tokens.
        if len(self.logit_std_history) >= 2:
            std_delta = self.logit_std_history[-1] - self.logit_std_history[0]
            if std_delta > self.logit_std_min_delta:
                self.patience_count = max(0, self.patience_count - 1)
                return self._result(
                    smoothed, is_best,
                    should_stop=False,
                    reason=(
                        f"[logit_std-growing] Δstd={std_delta:.4f}"
                        f" > {self.logit_std_min_delta}"
                    ),
                )

        # ── All gates passed → STOP ───────────────────────────────────────
        # Compute rel_improvement for the stop message (safe: best_raw is finite here).
        rel_final = (self.best_raw - current_val) / max(abs(self.best_raw), 1e-8)
        reason = (
            f"patience={self.patience_count}/{self.patience}  |  "
            f"val={current_val:.4f}  best={self.best_raw:.4f}  "
            f"rel_Δ={rel_final:.5f} < {self.min_delta}  |  "
            f"trend_slope={self._linear_slope(self.val_history):.5f}  |  "
            f"logit_std_Δ={self.logit_std_history[-1] - self.logit_std_history[0]:.4f}"
            if len(self.logit_std_history) >= 2
            else (
                f"patience={self.patience_count}/{self.patience}  |  "
                f"val={current_val:.4f}  best={self.best_raw:.4f}  "
                f"rel_Δ={rel_final:.5f} < {self.min_delta}"
            )
        )
        return self._result(smoothed, is_best, should_stop=True, reason=reason)

    # ── serialisation ─────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Return all mutable state for checkpoint persistence."""
        return {
            "best_raw":           self.best_raw,
            "patience_count":     self.patience_count,
            "val_history":        list(self.val_history),
            "logit_std_history":  list(self.logit_std_history),
            "ema_val":            self.ema_val,
            "_n_evals":           self._n_evals,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Restore mutable state from a checkpoint dict (forward-compatible)."""
        self.best_raw          = d.get("best_raw",          float("inf"))
        self.patience_count    = d.get("patience_count",    0)
        self.val_history       = list(d.get("val_history",  []))
        self.logit_std_history = list(d.get("logit_std_history", []))
        self.ema_val           = d.get("ema_val",           None)
        self._n_evals          = d.get("_n_evals",          0)
        # Legacy key from earlier checkpoint format — silently ignored.
        # ("best_smoothed" was the old EMA-based best; no longer used.)

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """
        Numerically stable OLS slope for a short list of floats.

        Returns 0.0 when fewer than 2 points.  No division-by-zero possible
        because the denominator is the sum of squared deviations from the
        mean index, which is always positive for n ≥ 2 distinct indices.
        """
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2                   for i in range(n))
        return num / max(den, 1e-10)

    def _result(
        self,
        smoothed:    float,
        is_best:     bool,
        should_stop: bool,
        reason:      str,
    ) -> Dict[str, Any]:
        return {
            "smoothed_val":   smoothed,
            "is_best":        is_best,
            "patience_count": self.patience_count,
            "should_stop":    should_stop,
            "reason":         reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class HiddenProjector(nn.Module):
    """
    Learnable linear projection from teacher hidden dim → student spatial dim.

    Replaces the non-learnable adaptive_avg_pool1d that previously destroyed
    structural information and blocked gradient flow through alignment.

    Architecture: Linear(D_teacher → D_student) with LayerNorm + residual gate.
    Initialized to approximate the avg-pool baseline (small random init).
    Frozen=False: gradient flows back through this projection → student learns
    to produce representations that align with GPT-2 after linear mapping.

    Thread-safe: one instance per (D_teacher, D_student) pair.
    """

    def __init__(self, d_teacher: int, d_student: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_teacher, d_student, bias=False)
        self.norm = nn.LayerNorm(d_student)
        # Small init — close to avg-pool baseline at start, improves gradually
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_teacher))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D_teacher]  →  [..., D_student]"""
        return self.norm(self.proj(x))


class TeacherDistillationLossV2(nn.Module):
    """
    Upgraded distillation loss for hyperbolic student + GPT-2 teacher.

    Loss terms
    ----------
    L_soft   : KL(student || teacher) with temperature scaling          [existing]
    L_hard   : CE(student, labels) with label smoothing                 [upgraded]
    L_hidden : cosine alignment student_tangent ↔ teacher_projected     [upgraded]
    L_radius : variance of geodesic radius → uniform manifold usage     [NEW]
    L_contrast: in-batch contrastive on tangent representations          [NEW]

    Key upgrades vs original
    ------------------------
    1. HiddenProjector replaces adaptive_avg_pool → gradient flows through alignment
    2. Full sequence alignment (no subsampling) — was losing 75% of gradient signal
    3. Radius regularization: prevents embeddings drifting to flat Euclidean region
    4. In-batch contrastive: forces structurally distinct sequences to differ on manifold
    5. Label smoothing 0.1 on CE: reduces overconfidence, improves calibration

    Geometric safety
    ----------------
    All operations on student hidden states go through log_map_zero → spatial,
    which guarantees the computation stays in Euclidean tangent space.
    No raw Lorentz addition or subtraction is performed outside substrate methods.
    Radius computed as geodesic distance from origin (substrate.dist) — exact.

    Zero imports from legacy cgt.*.
    """

    def __init__(
        self,
        substrate: LorentzSubstrateV2,
        temperature: float = 2.0,
        alpha: float = 0.5,
        lambda_hidden: float = 0.1,
        # New geometric loss weights
        lambda_radius: float = 0.01,
        lambda_contrast: float = 0.05,
        label_smoothing: float = 0.1,
        # Teacher hidden dim (GPT-2 = 768)
        teacher_hidden_dim: int = 768,
        # Last-k token pooling for contrastive loss (k=1 = last token only).
        # k=1: GPT-style — last position has full causal context.
        # k>1: mean of last k positions — more robust if last token overfits to EOS.
        # k=4: recommended default — smooths local noise, captures causal context.
        contrast_pool_k: int = 4,
    ) -> None:
        super().__init__()
        self.substrate          = substrate
        self.temperature        = temperature
        self.alpha              = alpha
        self.lambda_hidden      = lambda_hidden
        self.lambda_radius      = lambda_radius
        self.lambda_contrast    = lambda_contrast
        self.label_smoothing    = label_smoothing
        self._contrast_pool_k   = contrast_pool_k

        # ── Upgrade A: Learnable projection (replaces avg_pool) ───────────
        # Only built when teacher_hidden_dim ≠ student_spatial_dim
        student_spatial_dim = substrate.n   # n_embd (spatial, without time coord)
        if teacher_hidden_dim != student_spatial_dim:
            self.teacher_proj: Optional[HiddenProjector] = HiddenProjector(
                d_teacher=teacher_hidden_dim,
                d_student=student_spatial_dim,
            )
        else:
            self.teacher_proj = None

    def _student_to_tangent(
        self, student_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Map student Lorentz-ambient hidden states → Euclidean spatial tangent.

        Input : [B, L, n+1]  (Lorentz ambient)
        Output: [B, L, n]    (spatial part of tangent at origin, float32)

        Safety check: if mean tangent norm → 0, training must stop.
        """
        B, L, D_s = student_hidden.shape
        ambient = self.substrate.n + 1

        if D_s == ambient:
            s_flat = student_hidden.reshape(B * L, D_s)
            v = self.substrate.log_map_zero(s_flat).to(student_hidden.dtype)
            s_spatial = v[:, 1:].reshape(B, L, self.substrate.n)   # [B, L, n]

            # Safety check: tangent norm → 0 means embeddings at origin
            tangent_norm = s_spatial.norm(dim=-1).mean().item()
            if tangent_norm < 1e-6:
                import warnings
                warnings.warn(
                    f"[TangentCollapse] mean tangent norm={tangent_norm:.2e} < 1e-6. "
                    f"Student embeddings have collapsed to manifold origin. "
                    f"Cosine/contrastive gradients are zero — training is stuck. "
                    f"Increase lambda_radius or reduce lambda_distill.",
                    RuntimeWarning, stacklevel=3,
                )
            return s_spatial

        # Already Euclidean spatial
        return student_hidden

    def _radius_loss(self, student_hidden: torch.Tensor) -> torch.Tensor:
        """
        Geodesic radius regularization.

        With proj() now clamping spatial L2 norm at max_spatial_norm=5.0,
        the structural radius explosion is prevented at the source.
        This loss acts as a fine-tuning signal to keep radius near target=2.0
        (healthy operating point for 128-dim hyperbolic space with K=1).

        Pure MSE: symmetric gradient — pushes down if too high, up if too low.
        No directional bias (the old -mean() term caused explosion).
        """
        B, L, D_s = student_hidden.shape
        ambient = self.substrate.n + 1

        if D_s != ambient:
            return torch.tensor(0.0, device=student_hidden.device,
                                dtype=student_hidden.dtype)

        K = self.substrate.K.to(student_hidden.device, student_hidden.dtype)
        sqrt_K = torch.sqrt(K)
        x0 = student_hidden[..., 0]
        arg = (sqrt_K * x0).clamp(min=1.0 + 1e-6)
        radius = torch.log(arg + torch.sqrt(arg**2 - 1.0 + 1e-15)) / sqrt_K

        radius_mean = radius.mean()
        radius_var  = radius.var()

        import warnings
        if radius_mean.item() < 0.1:
            warnings.warn(
                f"[RadiusCollapse] mean_radius={radius_mean.item():.4f} < 0.1. "
                f"Embeddings collapsed to origin.",
                RuntimeWarning, stacklevel=3,
            )
        elif radius_mean.item() > 6.0:
            warnings.warn(
                f"[RadiusExplosion] mean_radius={radius_mean.item():.4f} > 6.0. "
                f"proj() spatial clamp may not be active — check lorentz_v2.py.",
                RuntimeWarning, stacklevel=3,
            )

        # FIX: target_radius 4.0 → 1.5
        #
        # ROOT CAUSE (validated via training_metrics.csv + code inspection):
        # At r=4.0:  sinh(4)=27.29, cosh(4)=27.31
        #   Minkowski inner <h,w>_M ≈ -745 ± 65.8  (128-dim concentration)
        #   logit_scale must grow ~11x to produce logit_std≈1
        #   With clamp(0.01,20) the optimizer freely inflates it → logit_std 7.36
        #   At inference (no temperature=3 suppression): word salad.
        #
        # At r=1.5: sinh(1.5)=2.13, cosh(1.5)=2.35
        #   Minkowski inner <h,w>_M ≈ -5.53 ± 0.40  (64x smaller variance)
        #   logit_scale stable near init (1/sqrt(128)≈0.088)
        #   logit_std numerically tractable throughout training AND inference.
        #
        # Additionally: max_spatial_norm=5.0 per-residual → max radius≈2.31
        #   Setting target=4.0 creates permanent opposing gradient (loss pulls up,
        #   clamp holds at 2.31). Setting target=1.5 < 2.31 resolves the conflict.
        #
        # Prevent Minkowski inner-product cancellation at large radii.
        target_radius = 1.5
        return F.mse_loss(radius.clamp(max=4.0), torch.full_like(radius, target_radius))

    def _contrastive_loss(
        self,
        s_spatial: torch.Tensor,           # [B, L, n]  student tangent
        t_projected: torch.Tensor,         # [B, L, n]  teacher projected
    ) -> torch.Tensor:
        """
        Lightweight in-batch contrastive loss (InfoNCE).

        FIX: replaced mean(dim=1) with norm-weighted aggregation.

        ROOT CAUSE of ctr=log(N) collapse:
          After LayerNorm with max_tangent_norm=1.5, all token hidden states
          have radius=1.5 and nearly-identical directions (l_hidden pulled them
          to the teacher projection, which shares a common axis across tokens).
          mean(128 nearly-parallel vectors) = common_direction for ALL sequences
          → cosine_sim(seq_i, seq_j) ≈ 1.0 → InfoNCE = log(B) = 2.773.

          Temperature reduction (0.07→0.05) alone cannot fix this because
          sim[i,j] ≈ 14.3 for ALL pairs — sharpening a uniform distribution
          still gives a uniform distribution.

        FIX — norm-weighted aggregation:
          w[b,t] = ||s_spatial[b,t]|| / Σ_t ||s_spatial[b,t]||
          Tokens with stronger tangent activation receive higher weight.
          Since tangent norms vary by position and sequence content
          (not all equal despite fixed manifold radius), this breaks the
          uniform-direction collapse and recovers sequence-level diversity.

        FIX — temperature 0.07 → 0.05:
          Amplifies residual angular differences that survive the aggregation.
          At τ=0.07 the softmax is already saturated once directions align;
          τ=0.05 gives finer resolution on the angular differences that matter.

        Safety check: if contrastive loss ≈ ln(B), embeddings have collapsed.
        """
        # Last-token pooling: the principled choice for causal (decoder-only) LMs.
        #
        # WHY NOT norm-weighted pooling:
        #   RiemannianLayerNormV2 with max_tangent_norm=1.5 forces ALL tokens to
        #   exactly radius=1.5 → tangent norm = sinh(1.5) = 2.129 for every token.
        #   Confirmed by telemetry: w_entropy = log(128) = 4.853 (maximum) always.
        #   => norm-weighted pooling degenerates to mean pooling with equal weights.
        #   Norm signal is dead by architecture. This cannot be fixed from pooling side.
        #
        # WHY last-token pooling:
        #   In a causal LM with attention mask, position L-1 has attended to ALL
        #   prior positions and carries maximal contextual information by construction.
        #   It is architecturally guaranteed to differ between sequences (different
        #   last tokens → different contextual representations regardless of norm).
        #   This is the standard aggregation for GPT-style sentence embeddings.
        #   Zero dependency on norm variation → immune to norm collapse.
        #
        # w_std / w_entropy telemetry now serves as norm-collapse diagnostics only
        # (not as a signal for pooling health). w_entropy ≈ log(L) is EXPECTED and
        # correct — it confirms uniform norms, which is fine with last-token pooling.
        # Last-k causal window pooling.
        #
        # Architecture:
        #   s_seq = normalize( mean( normalize(s_t) for t in [L-k, L) ) )
        #
        # Two-stage normalization:
        #   1. Per-token direction normalization (F.normalize per position):
        #      removes magnitude entirely — only direction contributes to pooling.
        #      Necessary because RiemannianLayerNorm forces all norms to sinh(1.5)=2.13,
        #      making raw magnitude uninformative. Normalizing here makes the pooling
        #      invariant to any future changes in the radius constraint.
        #   2. Sequence-level normalization (F.normalize on the pooled vector):
        #      ensures unit-norm input to the InfoNCE similarity computation.
        #
        # k=4 (default): mean of last 4 positions — smooths over position-final
        #   noise while preserving causal context. Empirically better trade-off
        #   than k=1 (too sensitive to last token) or k>>8 (dilutes context).
        k = self._contrast_pool_k

        # Directional Causal Pooling (DCP):
        #   1. Slice last-k positions (causal window)
        #   2. Normalize per-token direction (magnitude-invariant, radius-agnostic)
        #   3. Mean within window → unit-norm sequence embedding
        # Normalizing only the k-token slice avoids computing directions for
        # L-k unused positions, and makes the magnitude-invariance explicit.
        s_dirs = F.normalize(s_spatial[:, -k:, :], dim=-1)          # [B, k, n]
        s_seq  = s_dirs.mean(dim=1)
        s_seq  = s_seq / (s_seq.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n]

        t_dirs = F.normalize(t_projected[:, -k:, :], dim=-1)        # [B, k, n]
        t_seq  = t_dirs.mean(dim=1)
        t_seq  = t_seq / (t_seq.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n]

        B = s_seq.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=s_seq.device, dtype=s_seq.dtype)

        sim = torch.mm(s_seq, t_seq.t()) / 0.05   # FIX: 0.07 → 0.05
        labels = torch.arange(B, device=s_seq.device)
        loss_s2t = F.cross_entropy(sim, labels)
        loss_t2s = F.cross_entropy(sim.t(), labels)
        loss = (loss_s2t + loss_t2s) / 2.0

        # Collapse detection: fires only when loss is significantly above max-entropy
        # and the model is past warmup. Two-tier threshold:
        #   - Before warmup (step=None): silent — ctr≈ln(B) from random init is normal.
        #   - After warmup: threshold = ln(B)+0.30.
        #     ln(B)+0.05 was too tight (fires constantly at init due to noise).
        #     ln(B)+0.30 means positives are noticeably worse than random negatives.
        # Warning is throttled to once per N calls to avoid log spam.
        import math as _math
        import warnings
        _step = getattr(self, '_global_step', None)
        _warn_every = getattr(self, '_ctr_warn_every', 50)   # warn at most every 50 fwd passes
        _warn_count = getattr(self, '_ctr_warn_count', 0)
        _warm = getattr(self, '_warmup_steps', 300)
        collapse_threshold = _math.log(B) + 0.30
        if (
            _step is not None
            and _step > _warm
            and loss.item() >= collapse_threshold
            and _warn_count % _warn_every == 0
        ):
            warnings.warn(
                f"[ContrastiveCollapse] ctr={loss.item():.4f} > ln({B})+0.30={collapse_threshold:.4f} "
                f"after warmup. Last-token embeddings not separating sequences. "
                f"Consider: λ_contrast ↑, λ_distill ↓, temperature ↑.",
                RuntimeWarning, stacklevel=3,
            )
        self._ctr_warn_count = _warn_count + 1
        return loss

    def forward(
        self,
        student_logits: torch.Tensor,       # [B, L, V]
        teacher_logits: torch.Tensor,       # [B, L, V]
        labels: torch.Tensor,               # [B, L]
        student_hidden: Optional[torch.Tensor] = None,   # [B, L, n+1]
        teacher_hidden: Optional[torch.Tensor] = None,   # [B, L, D_teacher]
    ) -> Dict[str, torch.Tensor]:
        device = student_logits.device
        V = student_logits.shape[-1]

        s   = student_logits[..., :-1, :].contiguous()   # [B, L-1, V]
        t   = teacher_logits[..., :-1, :].contiguous()
        lbl = labels[..., 1:].contiguous()               # [B, L-1]

        # ── L_soft: KL divergence with temperature ────────────────────────
        s_log_p = F.log_softmax(s / self.temperature, dim=-1)
        t_p     = F.softmax(t / self.temperature, dim=-1)
        l_soft  = F.kl_div(
            s_log_p.reshape(-1, V), t_p.reshape(-1, V), reduction="batchmean"
        ) * (self.temperature ** 2)

        # ── L_hard: CE with label smoothing ──────────────────────────────
        # Upgrade B: label_smoothing=0.1 reduces overconfidence and
        # improves calibration. In practice gains 0.1-0.3 val PPL for small models.
        l_hard = F.cross_entropy(
            s.reshape(-1, V), lbl.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

        l_out = self.alpha * l_soft + (1.0 - self.alpha) * l_hard

        # ── L_hidden, L_radius, L_contrast ────────────────────────────────
        l_hidden   = torch.tensor(0.0, device=device, dtype=student_logits.dtype)
        l_radius   = torch.tensor(0.0, device=device, dtype=student_logits.dtype)
        l_contrast = torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        if student_hidden is not None:
            # ── Upgrade C: Radius regularization ─────────────────────────
            if self.lambda_radius > 0:
                l_radius = self._radius_loss(student_hidden)

            if teacher_hidden is not None:
                # ── Student → tangent space (geometrically correct) ───────
                s_spatial = self._student_to_tangent(student_hidden)  # [B, L, n]
                B, L, n   = s_spatial.shape

                # ── Upgrade A: Learnable projection (replaces avg_pool) ───
                t_spatial = teacher_hidden.to(s_spatial.dtype)        # [B, L, D_t]
                if self.teacher_proj is not None:
                    t_proj = self.teacher_proj(
                        t_spatial.reshape(B * L, -1)
                    ).reshape(B, L, n)                                 # [B, L, n]
                else:
                    t_proj = t_spatial  # already correct dim

                # ── L_hidden: cosine on FULL sequence (no subsampling) ────
                # Upgrade: removed 32-token subsampling → 4x more gradient
                if self.lambda_hidden > 0:
                    s_norm = F.normalize(s_spatial.reshape(-1, n), dim=-1)
                    t_norm = F.normalize(t_proj.reshape(-1, n), dim=-1)
                    l_hidden = (1.0 - (s_norm * t_norm).sum(dim=-1)).mean()

                # ── Upgrade D: In-batch contrastive ──────────────────────
                if self.lambda_contrast > 0:
                    l_contrast = self._contrastive_loss(s_spatial, t_proj)

        total = (
            l_out
            + self.lambda_hidden   * l_hidden
            + self.lambda_radius   * l_radius
            + self.lambda_contrast * l_contrast
        )

        return {
            "total":      total,
            "l_soft":     l_soft,
            "l_hard":     l_hard,
            "l_hidden":   l_hidden,
            "l_radius":   l_radius,
            "l_contrast": l_contrast,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillationConfigV2:
    """
    Config for DistillationTrainerV2.

    Config Precedence Rule (all aliased fields in __post_init__)
    -------------------------------------------------------------
        if canonical_value != default:
            use canonical_value
        elif alias_value is not None:
            use alias_value
        else:
            use default

    Canonical fields
    ----------------
    alpha, temperature, lambda_distill, lambda_hidden,
    lambda_radius, lambda_contrast, label_smoothing,
    learning_rate, weight_decay, max_steps, warmup_steps, gradient_clip,
    checkpoint_every, eval_every, log_every, keep_last_n_checkpoints,
    lr_floor, teacher_hidden_dim

    Legacy aliases
    --------------
    lr             → learning_rate
    n_steps        → max_steps
    warmup         → warmup_steps

    New fields (v3)
    ---------------
    lambda_radius    : weight for geodesic radius variance regularization
    lambda_contrast  : weight for in-batch contrastive alignment loss
    label_smoothing  : label smoothing for CE loss (0.0 = original behaviour)
    lr_floor         : minimum LR multiplier (was hardcoded 0.1)
    teacher_hidden_dim: GPT-2 hidden dim for projection (default 768)
    adaptive_lambda  : if True, linearly warm up lambda_hidden 0→target over warmup_steps
    """

    # ── canonical ──────────────────────────────────────────────────────────
    alpha: float                    = 0.5
    temperature: float              = 3.0
    lambda_distill: float           = 0.3
    lambda_hidden: float            = 0.15
    # New geometric loss weights
    lambda_radius: float            = 0.05
    lambda_contrast: float          = 0.05
    label_smoothing: float          = 0.1
    learning_rate: float            = 3e-4
    weight_decay: float             = 0.01
    max_steps: int                  = 20000
    warmup_steps: int               = 300
    gradient_clip: float            = 1.0
    # ── early stopping (V3 — dual-EMA, noise-aware, regime-sensitive) ────
    early_stopping_patience:        int   = 10     # base patience limit
    early_stopping_min_delta:       float = 0.005  # relative fast-EMA threshold (0.5 %)
    ema_beta:                       float = 0.9    # slow EMA (display + phase)
    ema_beta_fast:                  float = 0.3    # fast EMA (stopping decisions)
    min_steps_before_stopping:      int   = 1500   # warmup guard
    trend_window:                   int   = 5      # window for noise / phase / local ref
    noise_mult:                     float = 2.0    # hold gate: improvement > noise_mult×noise
    plateau_threshold:              int   = 5      # min stagnant evals before stop fires
    logit_std_min_delta:            float = 0.1    # logit_std growth gate
    phase3_patience_multiplier:     float = 1.0    # patience scale in Phase 3
    checkpoint_every: int           = 1000
    eval_every: int                 = 500
    log_every: int                  = 100
    keep_last_n_checkpoints: int    = 3
    # Bonus: configurable LR floor and curriculum settings
    lr_floor: float                 = 0.05   # was hardcoded 0.1; 0.05 = 1.5e-5 final
    lr_squeeze_after: int           = 0      # step to activate phase-2 cosine (0 = auto: 20% of max_steps)
    teacher_hidden_dim: int         = 768    # GPT-2 hidden dim
    adaptive_lambda: bool           = True   # warm up lambda_hidden gradually

    # ── Degenerate Equilibrium intervention ─────────────────────────────
    # Action taken when DEGENERATE_EQ regime is detected.
    # Audit finding: drift migrates to vocab embeddings (Euclidean, unconstrained).
    # Interventions target that channel specifically.
    #
    #  'none'             : log only, no action (Variant A / baseline)
    #  'stop'             : stop training immediately (Variant B)
    # ── Degenerate Equilibrium intervention (DegEqController) ────────────
    # See DegEqController docstring for full mechanism documentation.
    # Combine actions with '+': e.g. 'reweight_loss+adaptive_temp'
    deg_eq_action: str               = 'none'   # 'none'|'stop'|'freeze_vocab'|'progressive_freeze'|'reweight_loss'|'adaptive_temp'

    # ── Riemannian natural gradient correction (Amari 1998) ──────────────
    riemannian_correct_vocab:   bool = True
    riemannian_correct_embed:   bool = True
    riemannian_correct_encoder: bool = True

    deg_eq_consecutive_required: int = 3        # confirmations before action fires
    deg_eq_freeze_interval: int      = 2        # evals between progressive freeze steps
    deg_eq_reweight_step: float      = 0.05     # alpha reduction per confirmation
    deg_eq_alpha_min: float          = 0.15     # minimum KL alpha bound
    deg_eq_temp_scale: float         = 1.10     # temperature multiplier per confirmation
    deg_eq_temp_max: float           = 2.5      # temperature upper bound

    # ── legacy aliases ─────────────────────────────────────────────────────
    lr: Optional[float]     = None   # → learning_rate
    n_steps: Optional[int]  = None   # → max_steps
    warmup: Optional[int]   = None   # → warmup_steps

    # ── sentinels ─────────────────────────────────────────────────────────
    _D_LR: float  = field(default=3e-4,  init=False, repr=False)
    _D_NS: int    = field(default=20000, init=False, repr=False)
    _D_WU: int    = field(default=500,   init=False, repr=False)

    def __post_init__(self) -> None:
        if self.learning_rate != self._D_LR:
            pass
        elif self.lr is not None:
            self.learning_rate = self.lr

        if self.max_steps != self._D_NS:
            pass
        elif self.n_steps is not None:
            self.max_steps = self.n_steps

        if self.warmup_steps != self._D_WU:
            pass
        elif self.warmup is not None:
            self.warmup_steps = self.warmup


# ─────────────────────────────────────────────────────────────────────────────
# Teacher
# ─────────────────────────────────────────────────────────────────────────────

class GPT2TeacherWrapperV2(nn.Module):
    """
    Frozen GPT-2 teacher for knowledge distillation.
    Requires `transformers` package.
    Zero imports from legacy cgt.*.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu") -> None:
        super().__init__()
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as e:
            raise ImportError(
                "GPT2TeacherWrapperV2 requires `transformers`. "
                "Run: pip install transformers"
            ) from e

        self.device = device
        self.model  = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.config = self.model.config

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out = self.model(
            input_ids,
            output_hidden_states=return_hidden,
            return_dict=True,
        )
        result: Dict[str, torch.Tensor] = {"logits": out.logits}
        if return_hidden and out.hidden_states is not None:
            result["hidden_states"] = out.hidden_states[-1]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Loss Balance Monitor + Meta-tuner
# ─────────────────────────────────────────────────────────────────────────────

class GradNormMeasure:
    """
    Measures actual per-loss gradient norms using autograd.grad on a probe parameter.

    Why real grad norms matter
    --------------------------
    Loss magnitude (used by LossBalancer) is a proxy, not the truth.
    Example of failure mode:
      - KL loss = 0.5 nats  (small magnitude, but very peaked teacher → HUGE gradients)
      - Contrastive loss = 2.5 nats (large magnitude, but flat landscape → tiny gradients)
      Using EMA(loss) would try to REDUCE KL weight and INCREASE contrastive weight.
      Using real grad norms would correctly diagnose that KL is already dominating.

    Algorithm
    ---------
    Every ``measure_every`` steps, runs an ISOLATED forward+backward micro-step
    (no weight updates, no side effects on training state):

    1. Forward pass on the CURRENT batch (student + teacher, no_grad on teacher).
    2. Compute individual loss tensors: l_soft, l_hidden, l_contrast.
    3. For each: autograd.grad(λ_i * loss_i, probe, retain_graph=True).
    4. Record ||gradient|| at probe parameter.
    5. Compute imbalance_ratio = max_norm / min_norm.
    6. Clean up gradients (model.zero_grad).

    Probe parameter
    ---------------
    ``encoder.final_norm.layer_norm.weight`` — the LayerNorm gamma of the
    final encoder layer (128 floats). Small enough for fast output, deep enough
    in the encoder to receive gradients from ALL loss terms.

    Cost
    ----
    3 extra autograd.grad calls (retain_graph=True) every ``measure_every``
    steps. Each traverses the full computation graph but produces only 128
    gradient floats. Overhead ≈ 3/measure_every of total backward cost.
    At measure_every=500: ~0.6% overhead.

    Integration with LossBalancer
    ------------------------------
    LossBalancer.update_from_grad_norms() can optionally incorporate
    the measured gradient norms to replace EMA(loss) as the balancing signal.
    When grad norms are available, they provide ground-truth contribution.
    When unavailable (measurement step skipped), EMA(loss) is used as fallback.
    """

    def __init__(self, measure_every: int = 500) -> None:
        self.measure_every = measure_every
        self.history: List[Dict] = []
        self._last_norms: Dict[str, float] = {}

    def _find_probe(self, student_model) -> Optional[Any]:
        """Return the final LayerNorm weight as probe, or any small param."""
        for name, p in student_model.named_parameters():
            if 'final_norm' in name and 'weight' in name and p.requires_grad:
                return p
        # Fallback: any LayerNorm weight
        for name, p in student_model.named_parameters():
            if 'norm' in name.lower() and 'weight' in name and p.requires_grad and p.numel() < 512:
                return p
        # Last resort: first available param with grad
        return next((p for p in student_model.parameters() if p.requires_grad), None)

    def measure(
        self,
        loss_tensors: Dict[str, Any],   # name → (lambda_weight, loss_tensor)
        student_model,
        step: int,
    ) -> Dict[str, float]:
        """
        Measure gradient norms for each loss at the probe parameter.

        Args:
            loss_tensors: dict of {name: (effective_lambda, loss_tensor)}
                          loss_tensor must still have a valid computation graph.
            student_model: the student model (used to find probe parameter).
            step: current training step.

        Returns: dict of {name: grad_norm, "imbalance_ratio": float}.
                 Empty dict if measurement is skipped this step.
        """
        if step % self.measure_every != 0:
            return {}

        probe = self._find_probe(student_model)
        if probe is None:
            return {}

        norms: Dict[str, float] = {}
        for name, (lam, loss_val) in loss_tensors.items():
            if loss_val is None:
                norms[name] = 0.0
                continue
            if isinstance(loss_val, torch.Tensor) and not loss_val.requires_grad:
                norms[name] = 0.0
                continue
            try:
                effective_loss = lam * loss_val if isinstance(loss_val, torch.Tensor) else 0.0
                if not isinstance(effective_loss, torch.Tensor):
                    norms[name] = 0.0
                    continue
                grad = torch.autograd.grad(
                    effective_loss, probe,
                    retain_graph=True, allow_unused=True,
                )[0]
                norms[name] = float(grad.norm().item()) if grad is not None else 0.0
            except Exception:
                norms[name] = 0.0

        active = {k: v for k, v in norms.items() if v > 1e-12}
        imbalance = max(active.values()) / min(active.values()) if len(active) >= 2 else 1.0
        norms["imbalance_ratio"] = imbalance

        record = {"step": step, **norms}
        self.history.append(record)
        self._last_norms = norms
        return norms


class LossBalancer:
    """
    Meta-tuner: automatically adjusts loss lambdas to equalize effective
    gradient contributions, complementing the AdaptiveTuner.

    Relationship to AdaptiveTuner
    -----------------------------
    AdaptiveTuner: *reactive* — fires when a regime symptom is detected
      (hid > 0.5, logit_std > 2.5, etc.). Responds to crises.

    LossBalancer: *proactive* — runs every eval, gently steers lambdas
      toward balance before a crisis develops.

    Together they form a two-layer control system:
      LossBalancer → coarse balance (runs always)
      AdaptiveTuner → fine crisis response (runs on regime alert)

    Algorithm (EMA-based, GradNorm-inspired)
    -----------------------------------------
    For each loss term i, track EMA of loss magnitude:
      ema_i = β * ema_i + (1-β) * loss_i

    Effective contribution:
      eff_i = λ_i * ema_i

    Target: all eff_i equal.
      target = mean(eff_i for i in active)

    Soft update (α = smoothing rate, default 0.3):
      λ_i_new = λ_i * (target / eff_i)^α

    Clipped to hard bounds and bounded by max_step_factor per update.

    Active terms (adjusted): l_hidden, l_contrast.
    Fixed terms (not adjusted): l_distill, temperature — these define the
      fundamental teacher signal and are controlled by AdaptiveTuner.

    Parameters
    ----------
    ema_beta : float
        EMA smoothing factor for loss tracking. Default 0.7.
        Lower = faster adaptation to loss changes.
    alpha : float
        Soft-update exponent. 0.3 = gentle, 1.0 = immediate.
    max_step_factor : float
        Maximum multiplicative change per update. Default 1.15 (±15%).
    balance_every : int
        How many evals between balance attempts. Default 2.
    """

    BOUNDS = {
        "lambda_hidden":   (0.05, 0.40),   # widened from 0.30 — allows real compensation
        "lambda_contrast": (0.03, 0.25),   # widened from 0.20 — allows real compensation
    }

    def __init__(
        self,
        loss_fn,
        ema_beta:        float = 0.7,
        alpha:           float = 0.3,
        max_step_factor: float = 1.15,
        balance_every:   int   = 2,
        warmup_guard:    int   = 500,
    ) -> None:
        self.loss_fn        = loss_fn
        self.ema_beta       = ema_beta
        self.alpha          = alpha
        self.max_step_factor= max_step_factor
        self.balance_every  = balance_every
        self.warmup_guard   = warmup_guard

        self._ema:        Dict[str, float] = {}
        self._eval_count: int              = 0
        self.balance_log: List[Dict]       = []
        self.skip_params: set              = set()   # params frozen by AdaptiveTuner during crisis

    def update_from_grad_norms(self, norms: Dict[str, float]) -> None:
        """
        Optionally replace EMA(loss) with real gradient norms for balancing signal.
        Called after GradNormMeasure.measure() when real norms are available.
        Grad norms are stored under synthetic EMA keys prefixed with 'gnorm_'.
        The balance() method prefers these when present.
        """
        for name, norm in norms.items():
            if name in ("imbalance_ratio",):
                continue
            self._ema[f"gnorm_{name}"] = norm

    def update_ema(self, loss_vals: Dict[str, float]) -> None:
        """Call every step with current loss values to keep EMA fresh."""
        for name, val in loss_vals.items():
            if name not in self._ema:
                self._ema[name] = val
            else:
                self._ema[name] = (self.ema_beta * self._ema[name]
                                   + (1.0 - self.ema_beta) * val)

    def balance(self, step: int) -> List[Dict]:
        """
        Called every eval. Adjusts lambda_hidden and lambda_contrast
        to equalize effective contributions.

        Returns list of applied adjustments (empty if no change needed).
        """
        self._eval_count += 1
        applied: List[Dict] = []

        if step < self.warmup_guard:
            return applied
        if self._eval_count % self.balance_every != 0:
            return applied

        # Map param name → (lambda getter, loss EMA key)
        terms = {
            "lambda_hidden":   ("l_hidden",   getattr(self.loss_fn, "lambda_hidden",   0.15)),
            "lambda_contrast": ("l_contrast",  getattr(self.loss_fn, "lambda_contrast", 0.10)),
        }

        # AdaptiveTuner priority: skip params frozen during active crisis
        terms_active = {
            k: v for k, v in terms.items()
            if k not in self.skip_params
        }
        if not terms_active:
            return applied

        # Use real grad norms if available (from GradNormMeasure), else EMA(loss)
        # Grad norm keys are stored as 'gnorm_{ema_key}' when measured.
        effs: Dict[str, float] = {}
        for param, (ema_key, lam) in terms_active.items():
            gnorm_key = f"gnorm_{ema_key}"
            if gnorm_key in self._ema and self._ema[gnorm_key] > 1e-10:
                # Ground-truth: use measured gradient norm directly
                effs[param] = self._ema[gnorm_key]
            else:
                # Fallback: EMA(loss) × lambda as proxy
                ema_val = self._ema.get(ema_key, None)
                if ema_val is None or ema_val < 1e-8:
                    continue
                effs[param] = lam * ema_val

        if len(effs) < 2:
            return applied

        target = sum(effs.values()) / len(effs)

        # Meta-variance: measures how far from equilibrium we are.
        # Logged for diagnostic purposes (not used in control directly).
        meta_var = sum((e - target) ** 2 for e in effs.values()) / len(effs)

        for param, eff in effs.items():
            if eff < 1e-10:
                continue

            # Soft update: λ_new = λ * (target/eff)^α
            scale = (target / eff) ** self.alpha
            scale = max(1.0 / self.max_step_factor,
                        min(self.max_step_factor, scale))
            lam_curr = getattr(self.loss_fn, param)
            lam_new  = lam_curr * scale
            lo, hi   = self.BOUNDS[param]
            lam_new  = float(max(lo, min(hi, lam_new)))

            if abs(lam_new - lam_curr) < 1e-5:
                continue

            setattr(self.loss_fn, param, lam_new)
            rec = {
                "step":    step,
                "param":   param,
                "before":  round(lam_curr, 5),
                "after":   round(lam_new,  5),
                "eff_before": round(eff, 5),
                "target":  round(target, 5),
                "meta_var": round(meta_var, 6),
                "signal":  "gnorm" if f"gnorm_{terms_active[param][0]}" in self._ema else "ema_loss",
            }
            self.balance_log.append(rec)
            applied.append(rec)

        return applied


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Tuner
# ─────────────────────────────────────────────────────────────────────────────

class DegEqController:
    """
    Post-Convergence Intervention Framework (Algorithm D v2).

    Triggered when the DEGENERATE_EQ regime is confirmed (rdc_ema > threshold,
    l_hidden low and stable). Applies one or more of three mechanisms that
    target the source of post-convergence drift: unconstrained Euclidean
    parameters (vocab embeddings, logit_scale) and the objective dynamics
    (KL weight, temperature) that incentivize continued radial scaling.

    Mechanisms
    ----------
    'none'               : log only, no action (Variant A / baseline).

    'stop'               : stop training immediately (Variant B).
                           Use to test whether post-DEQ steps add value.

    'freeze_vocab'       : freeze lm_head + logit_scale (Variant C).
                           Blocks drift at its Euclidean source.
                           Angular learning (encoder) continues.
                           Prediction: logit_std ↓ or stable after freeze.

    'progressive_freeze' : freeze encoder layers bottom-up, one per
                           freeze_interval evals after DEQ, ending with lm_head.
                           Softer than freeze_vocab — prevents abrupt gradient
                           disruption. Use when freeze_vocab degrades PPL.

    'reweight_loss'      : reduce config.alpha (KL blend ratio) by reweight_step
                           each time DEQ is detected, bounded by alpha_min.
                           KL is the gradient driving radial scale; softening
                           it reduces the incentive to grow vocab norms.

    'adaptive_temp'      : increase temperature by temp_scale each DEQ
                           confirmation, bounded by temp_max.
                           Softer KL targets → smaller KL gradient magnitudes.
                           Distinct from AdaptiveTuner: this targets the
                           post-convergence equilibrium, not crisis regimes.

    Mechanisms can be combined with '+':
        deg_eq_action = 'reweight_loss+adaptive_temp'

    Configuration (DistillationConfigV2 fields)
    --------------------------------------------
    deg_eq_action                : str   = 'none'
    deg_eq_consecutive_required  : int   = 3      (confirmations before action)
    deg_eq_freeze_interval       : int   = 2      (evals between progressive freeze steps)
    deg_eq_reweight_step         : float = 0.05   (alpha reduction per confirmation)
    deg_eq_alpha_min             : float = 0.15   (minimum KL weight)
    deg_eq_temp_scale            : float = 1.10   (temperature multiplier per confirmation)
    deg_eq_temp_max              : float = 2.5    (temperature upper bound)

    Relationship to AdaptiveTuner
    -----------------------------
    AdaptiveTuner: reactive to training crises (KL_DOMINANCE, OVERCONFIDENT, etc.)
    DegEqController: reactive to confirmed post-convergence equilibrium state.
    They operate on different timescales and different signals.
    """

    def __init__(self, config, loss_fn, student_model) -> None:
        self.config   = config
        self.loss_fn  = loss_fn
        self.student  = student_model

        self._count          = 0      # consecutive DEQ detections
        self._triggered      = False  # first intervention applied
        self._freeze_idx     = 0      # index into progressive freeze schedule
        self._freeze_timer   = 0      # evals since last progressive freeze step
        self.intervention_log: List[Dict] = []

        # Build progressive freeze schedule (bottom encoder → top encoder → lm_head)
        self._prog_schedule  = self._build_freeze_schedule()

    def _build_freeze_schedule(self) -> List[str]:
        """Layer names in freeze order: shallow encoder first, lm_head last."""
        schedule = []
        for name, _ in self.student.named_parameters():
            for tag in ['embed', 'layer.0', 'layer.1', 'layer.2', 'layer.3']:
                if tag in name and name not in schedule:
                    schedule.append(name)
                    break
        # Always end with lm_head / logit_scale
        for name, _ in self.student.named_parameters():
            if ('lm_head' in name or 'logit_scale' in name) and name not in schedule:
                schedule.append(name)
        return schedule

    def _freeze_params(self, names: List[str], optimizer) -> int:
        """Freeze named parameters, rebuild optimizer param group. Returns count."""
        frozen = 0
        name_set = set(names)
        for name, p in self.student.named_parameters():
            if name in name_set and p.requires_grad:
                p.requires_grad = False
                frozen += p.numel()
        # Rebuild optimizer param groups preserving scheduler state
        active = [p for p in self.student.parameters() if p.requires_grad]
        optimizer.param_groups[0] = {
            **optimizer.param_groups[0],
            'params': active,
        }
        return frozen

    def step(
        self,
        regime_code: str,
        step: int,
        val_hist: List[Dict],
        optimizer,
    ) -> List[Dict]:
        """
        Called after each eval. Checks for DEGENERATE_EQ and applies configured
        interventions when confirmation threshold is met.

        Returns list of applied intervention dicts (empty if none applied).
        """
        if self._triggered and 'progressive_freeze' not in getattr(self.config, 'deg_eq_action', ''):
            return []   # all non-progressive actions are one-shot

        actions   = set(getattr(self.config, 'deg_eq_action', 'none').split('+'))
        required  = getattr(self.config, 'deg_eq_consecutive_required', 3)
        applied: List[Dict] = []

        if "DEGENERATE_EQ" in regime_code:
            self._count += 1
        else:
            self._count = 0
            return applied

        if self._count < required:
            return applied

        # ── 'stop' ─────────────────────────────────────────────────────────
        if 'stop' in actions and not self._triggered:
            self._triggered = True
            rec = {"step": step, "action": "stop", "count": self._count}
            applied.append(rec)
            self.intervention_log.append(rec)
            print(f"\n  🛑 DegEq STOP  step={step}  rdc_ema confirmed × {self._count}")
            if val_hist:
                val_hist[-1]["deg_eq_step"] = step
            # Caller must set self.stop = True
            return applied

        # ── 'freeze_vocab' ─────────────────────────────────────────────────
        if 'freeze_vocab' in actions and not self._triggered:
            self._triggered = True
            vocab_names = [n for n, _ in self.student.named_parameters()
                           if 'lm_head' in n or 'logit_scale' in n]
            frozen = self._freeze_params(vocab_names, optimizer)
            rec = {"step": step, "action": "freeze_vocab",
                   "frozen_params": frozen, "count": self._count}
            applied.append(rec)
            self.intervention_log.append(rec)
            print(f"\n  🧊 DegEq FREEZE_VOCAB  step={step}  frozen={frozen:,} params")
            print(f"     Predict: logit_std ↓, PPL ≈ baseline")
            if val_hist:
                val_hist[-1]["deg_eq_step"] = step

        # ── 'progressive_freeze' ────────────────────────────────────────────
        if 'progressive_freeze' in actions:
            self._triggered = True
            self._freeze_timer += 1
            interval = getattr(self.config, 'deg_eq_freeze_interval', 2)
            if self._freeze_timer >= interval and self._freeze_idx < len(self._prog_schedule):
                name = self._prog_schedule[self._freeze_idx]
                frozen = self._freeze_params([name], optimizer)
                self._freeze_idx  += 1
                self._freeze_timer = 0
                rec = {"step": step, "action": "progressive_freeze",
                       "layer": name, "frozen_params": frozen,
                       "freeze_idx": self._freeze_idx}
                applied.append(rec)
                self.intervention_log.append(rec)
                print(f"  🧊 DegEq PROG_FREEZE  step={step}  "
                      f"layer={name[:40]}  ({self._freeze_idx}/{len(self._prog_schedule)})")
            if val_hist:
                val_hist[-1].setdefault("deg_eq_step", step)

        # ── 'reweight_loss' ─────────────────────────────────────────────────
        if 'reweight_loss' in actions:
            alpha_cur = self.config.alpha
            alpha_min = getattr(self.config, 'deg_eq_alpha_min', 0.15)
            step_size = getattr(self.config, 'deg_eq_reweight_step', 0.05)
            alpha_new = max(alpha_min, alpha_cur - step_size)
            if alpha_new < alpha_cur:
                self.config.alpha   = alpha_new
                self.loss_fn.alpha  = alpha_new   # sync to loss fn
                rec = {"step": step, "action": "reweight_loss",
                       "alpha_before": round(alpha_cur, 4),
                       "alpha_after":  round(alpha_new, 4)}
                applied.append(rec)
                self.intervention_log.append(rec)
                print(f"  ⚖️  DegEq REWEIGHT  step={step}  alpha: {alpha_cur:.3f} → {alpha_new:.3f}")

        # ── 'adaptive_temp' ─────────────────────────────────────────────────
        if 'adaptive_temp' in actions:
            t_cur   = float(self.loss_fn.temperature)
            t_scale = getattr(self.config, 'deg_eq_temp_scale', 1.10)
            t_max   = getattr(self.config, 'deg_eq_temp_max',   2.5)
            t_new   = min(t_max, t_cur * t_scale)
            if t_new > t_cur + 0.01:
                self.loss_fn.temperature = t_new
                rec = {"step": step, "action": "adaptive_temp",
                       "temp_before": round(t_cur, 4),
                       "temp_after":  round(t_new, 4)}
                applied.append(rec)
                self.intervention_log.append(rec)
                print(f"  🌡️  DegEq TEMP  step={step}  T: {t_cur:.3f} → {t_new:.3f}")

        if applied and val_hist:
            val_hist[-1]["deg_eq_interventions"] = applied

        return applied


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Tuner
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTuner:
    """
    Closed-loop hyperparameter controller driven by regime codes.

    Detects regime codes emitted by the training health check and applies
    bounded, logged adjustments to live training parameters — no restart needed.

    Stability guarantees (Lyapunov-bounded)
    ----------------------------------------
    The system is stable in the sense that all parameter trajectories are
    strictly bounded regardless of training dynamics:

    1. Hard bounds: every parameter has a [lo, hi] constraint.
       No adjustment can ever push a parameter outside this range.

    2. Finite adjustments: ``max_adjusts_per_param`` caps the total number
       of interventions on any single parameter over the run.
       Worst-case temperature trajectory: T_init × 1.2^3 ≤ 2.5 (hard upper).

    3. Cooldown: at least ``cooldown_evals`` evaluations must pass between
       any two adjustments. Prevents rapid oscillation.

    4. Anti-oscillation: a regime code different from the last applied regime
       must be observed ``consecutive_required`` times consecutively before
       it triggers an adjustment. This filters regime noise (e.g., a single
       UNDERCONFIDENT eval following a KL_DOMINANCE adjustment) without
       blocking genuine sustained signals.

       Same-regime continuations are allowed immediately — if the model is
       still showing KL_DOMINANCE two adjustments later, the signal is real.

    Example cycle analysis:
      - KL_DOMINANCE fires: T 1.0 → 1.2
      - Next eval: UNDERCONFIDENT (single)  → not acted on (count 1/2)
      - Next eval: UNDERCONFIDENT again     → acted on (count 2/2): T 1.2 → 1.08
      - Net drift: T × 1.2 × 0.9 = T × 1.08 per cycle
      - Hard bound at 2.5 + max_adjusts=3 terminates any drift.

    Design principles
    -----------------
    Conservative: cooldown, max adjustments, hard bounds, warmup skip.
    Transparent: every adjustment is appended to ``tuning_log`` and attached
    to ``val_hist[-1]["auto_tune_log"]`` for post-hoc CSV analysis.
    """

    POLICY: Dict[str, List[Dict]] = {
        "KL_DOMINANCE":  [
            {"param": "temperature",     "op": "mul", "delta": 1.20, "lo": 0.8, "hi": 2.5},
            {"param": "lambda_distill",  "op": "add", "delta":-0.05, "lo": 0.2, "hi": 0.5},
        ],
        "UNDERCONFIDENT": [
            {"param": "temperature",     "op": "mul", "delta": 0.90, "lo": 0.8, "hi": 2.5},
        ],
        "OVERCONFIDENT": [
            {"param": "temperature",     "op": "mul", "delta": 1.10, "lo": 0.8, "hi": 2.5},
        ],
        "PPL_STAGNANT": [
            {"param": "temperature",     "op": "mul", "delta": 0.90, "lo": 0.8, "hi": 2.5},
            {"param": "lambda_distill",  "op": "add", "delta": 0.03, "lo": 0.2, "hi": 0.5},
        ],
        "CTR_COLLAPSE": [
            {"param": "lambda_contrast", "op": "add", "delta":-0.02, "lo": 0.02,"hi": 0.15},
        ],
        # DEGENERATE_EQ: explicitly NO tuner actions.
        # This is an equilibrium state, not a crisis. Hyperparameter adjustments
        # cannot fix it because it requires structural change (scale-invariant objective
        # or vocab embedding regularization). Attempting to tune it would waste
        # adjustment budget on a structurally intractable regime.
        # It is logged (regime_code, rdc_ema) for post-hoc analysis only.
        "DEGENERATE_EQ": [],
    }

    def __init__(
        self,
        loss_fn,
        config,
        warmup_guard:          int = 300,
        cooldown_evals:        int = 3,
        max_adjusts_per_param: int = 3,
        consecutive_required:  int = 2,
    ) -> None:
        self.loss_fn               = loss_fn
        self.config                = config
        self.warmup_guard          = warmup_guard
        self.cooldown_evals        = cooldown_evals
        self.max_adjusts_per_param = max_adjusts_per_param
        self.consecutive_required  = consecutive_required

        self._evals_since_tune:   int            = cooldown_evals
        self._adjust_counts:      Dict[str, int]  = {}
        self._last_applied_code:  Optional[str]   = None   # last code that triggered an action
        self._pending_code:       Optional[str]   = None   # candidate code (different from last)
        self._pending_count:      int             = 0      # consecutive detections of pending
        self.tuning_log:          List[Dict]      = []

    def _get(self, param: str) -> float:
        if param == "temperature":
            return float(self.loss_fn.temperature)
        if param == "lambda_distill":
            return float(self.config.lambda_distill)
        if param == "lambda_contrast":
            return float(self.loss_fn.lambda_contrast)
        raise KeyError(f"AdaptiveTuner: unknown param '{param}'")

    def _set(self, param: str, value: float) -> None:
        if param == "temperature":
            self.loss_fn.temperature = value
        elif param == "lambda_distill":
            self.config.lambda_distill = value
        elif param == "lambda_contrast":
            self.loss_fn.lambda_contrast = value
        else:
            raise KeyError(f"AdaptiveTuner: unknown param '{param}'")

    def step(
        self,
        regime_code:  str,
        current_step: int,
        val_hist:     List[Dict],
    ) -> List[Dict]:
        """
        Evaluate regime_code and apply adjustments if criteria are met.

        Anti-oscillation logic:
          - If ``regime_code == _last_applied_code``: the same problem persists.
            Act immediately (same-pattern continuation, not noise).
          - If ``regime_code != _last_applied_code``: require
            ``consecutive_required`` detections in a row before acting.
            A single-eval opposite signal (noise) is ignored.

        Returns list of applied adjustments, also attached to
        ``val_hist[-1]["auto_tune_log"]``.
        """
        self._evals_since_tune += 1
        applied: List[Dict] = []

        if current_step < self.warmup_guard:
            return applied
        if self._evals_since_tune < self.cooldown_evals:
            return applied
        if regime_code == "OK" or not regime_code:
            self._pending_code  = None
            self._pending_count = 0
            return applied

        # Anti-oscillation gate: new/different regime must persist
        if regime_code != self._last_applied_code:
            if regime_code == self._pending_code:
                self._pending_count += 1
            else:
                self._pending_code  = regime_code
                self._pending_count = 1

            if self._pending_count < self.consecutive_required:
                print(
                    f"  👀 Regime [{regime_code}] detected  "
                    f"({self._pending_count}/{self.consecutive_required} needed to act)"
                )
                return applied
        # else: same as last applied → act immediately, no confirmation needed

        # Process each code in a composite regime (e.g., "KL_DOMINANCE|PPL_STAGNANT")
        for code in regime_code.split("|"):
            code = code.strip()
            if code not in self.POLICY:
                continue

            for rule in self.POLICY[code]:
                param = rule["param"]
                count = self._adjust_counts.get(param, 0)
                if count >= self.max_adjusts_per_param:
                    continue

                old_val = self._get(param)
                if rule["op"] == "mul":
                    new_val = old_val * rule["delta"]
                else:
                    new_val = old_val + rule["delta"]
                new_val = float(max(rule["lo"], min(rule["hi"], new_val)))

                if abs(new_val - old_val) < 1e-6:
                    continue   # already at bound

                self._set(param, new_val)
                self._adjust_counts[param] = count + 1

                record = {
                    "step":   current_step,
                    "code":   code,
                    "param":  param,
                    "before": round(old_val, 5),
                    "after":  round(new_val, 5),
                    "count":  self._adjust_counts[param],
                }
                self.tuning_log.append(record)
                applied.append(record)

                print(
                    f"  🔧 AutoTune [{code}]  {param}: "
                    f"{old_val:.4f} → {new_val:.4f}  "
                    f"(adj #{self._adjust_counts[param]}/{self.max_adjusts_per_param})"
                )

        if applied:
            self._evals_since_tune  = 0
            self._last_applied_code = regime_code
            self._pending_code      = None
            self._pending_count     = 0
            if val_hist:
                val_hist[-1]["auto_tune_log"] = applied

        return applied


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DistillationTrainerV2:
    """
    Training loop for GPT-2 → SafeHyperbolicModel distillation.

    Compatible with SafeHyperbolicModel or HyperbolicTransformerV2.
    Supports LM-only mode (teacher=None, lambda_distill=0).
    Zero imports from legacy cgt.*.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[GPT2TeacherWrapperV2],
        config: DistillationConfigV2,
        tokenizer: Any,
        checkpoint_dir: Path,
        device: str = "cpu",
    ) -> None:
        self.student        = student
        self.teacher        = teacher
        self.config         = config
        self.tokenizer      = tokenizer
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        def _lr_lambda(step: int) -> float:
            """
            Two-phase LR schedule with automatic squeeze.

            Phase 1 (warmup → lr_squeeze_after):
              Cosine decay from lr_max → lr at squeeze point.
              Standard behaviour: model learns coarse structure.

            Phase 2 (lr_squeeze_after → max_steps):
              Aggressive cosine from lr_at_squeeze → eta_min.
              Triggered when the model has converged structurally
              (PPL stable, gradients small).
              Accelerates fine-grained optimization for the last
              PPL squeeze. Without this, the standard cosine at
              step 2000 is still at ~93% of lr_max — too high for
              noise-limited refinement.

            Configuration:
              lr_floor         : absolute minimum LR multiplier
              lr_squeeze_after : step at which phase 2 activates
                                 (default: 0.2 × max_steps)
            """
            squeeze_after = getattr(config, 'lr_squeeze_after',
                                    int(0.20 * config.max_steps))
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)

            if step < squeeze_after:
                # Phase 1: standard cosine
                progress = (step - config.warmup_steps) / max(
                    1, config.max_steps - config.warmup_steps
                )
                return max(config.lr_floor, 0.5 * (1.0 + math.cos(math.pi * progress)))

            # Phase 2: restart cosine from lr_at_squeeze → lr_floor
            # lr_at_squeeze is the multiplier at the squeeze entry point
            prog_squeeze = (squeeze_after - config.warmup_steps) / max(
                1, config.max_steps - config.warmup_steps
            )
            lr_at_squeeze = max(config.lr_floor, 0.5 * (1.0 + math.cos(math.pi * prog_squeeze)))
            remaining = config.max_steps - squeeze_after
            progress2 = (step - squeeze_after) / max(1, remaining)
            squeezed = config.lr_floor + 0.5 * (lr_at_squeeze - config.lr_floor) * (
                1.0 + math.cos(math.pi * progress2)
            )
            return max(config.lr_floor, squeezed)

        # Build substrate from student
        substrate = self._get_substrate(student)
        self.loss_fn = TeacherDistillationLossV2(
            substrate          = substrate,
            temperature        = config.temperature,
            alpha              = config.alpha,
            lambda_hidden      = config.lambda_hidden,
            lambda_radius      = config.lambda_radius,
            lambda_contrast    = config.lambda_contrast,
            label_smoothing    = config.label_smoothing,
            teacher_hidden_dim = config.teacher_hidden_dim,
        ).to(self.device)  # FIX: teacher_proj weights must be on same device as inputs

        # ── Upgrade A: Teacher projector is a trainable module ─────────────
        # Its parameters are separate from the student — use same optimizer
        # with a slightly higher LR (projection layer is small, converges fast)
        if self.loss_fn.teacher_proj is not None:
            proj_params = list(self.loss_fn.teacher_proj.parameters())
            self.optimizer.add_param_group({
                "params":       proj_params,
                "lr":           config.learning_rate * 2.0,   # faster convergence
                "weight_decay": 0.0,   # no decay on projection layer
            })

        # FIX: scheduler MUST be created after all add_param_group() calls.
        # LambdaLR captures param group count at construction; adding groups
        # afterward causes zip(strict=True) to fail in scheduler.step().
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _lr_lambda
        )

        self.step      = 0
        self.best_val  = float("inf")
        self.patience  = 0
        self._rdc_beta = 0.95   # EMA decay for RDC proxy
        self._rdc_ema  = 0.0    # EMA of RDC proxy; used for Degenerate Equilibrium detection
        # Scientific note on RDC and Degenerate Equilibrium:
        # The system does NOT exhibit runaway logit growth in late training (Phase 3).
        # Instead it reaches a "Degenerate Equilibrium" (audit finding, 2026-04):
        #   ∇_ang → 0  (semantic convergence: l_hidden stabilizes)
        #   radial scale → high but STABLE (logit_std plateaus, not diverges)
        # The drift is NOT in hidden states (bounded by max_tangent_norm=1.5 in LayerNorm)
        # but in vocab embeddings (Euclidean, no manifold constraint).
        # rdc_proxy = logit_std / (l_hidden + ε) captures this: when angular learning
        # is complete and radial scale dominates, rdc_proxy → high constant.
        # rdc_ema (β=0.95) smooths per-step noise for reliable regime detection.
        self._deg_eq_count     = 0      # consecutive DEGENERATE_EQ detections this run
        self._deg_eq_triggered = False  # whether intervention has been applied
        self.patience  = 0
        self.stop      = False
        self.train_hist: List[Dict] = []
        self.val_hist:   List[Dict] = []
        self._ckpts:     List[Path] = []

        # ── AdaptiveTuner — closed-loop regime-driven hyperparameter control ──
        self.adaptive_tuner = AdaptiveTuner(
            loss_fn        = self.loss_fn,
            config         = self.config,
            warmup_guard   = config.warmup_steps,
            cooldown_evals = 3,
            max_adjusts_per_param = 3,
        )

        # ── LossBalancer — proactive EMA-based lambda equalization ────────
        self.loss_balancer = LossBalancer(
            loss_fn      = self.loss_fn,
            warmup_guard = config.warmup_steps + config.eval_every,
        )

        # ── GradNormMeasure — ground-truth gradient contribution audit ────
        self.grad_norm_measure = GradNormMeasure(
            measure_every = config.checkpoint_every,
        )

        # ── DegEqController — post-convergence intervention framework ─────
        self.deg_eq_ctrl = DegEqController(
            config        = self.config,
            loss_fn       = self.loss_fn,
            student_model = self.student,
        )

        # ── EarlyStoppingV3 — dual-EMA reference tracking ────────────────
        self.early_stopper = EarlyStoppingV3(
            patience                   = config.early_stopping_patience,
            ema_beta                   = config.ema_beta,
            ema_beta_fast              = config.ema_beta_fast,
            min_delta                  = config.early_stopping_min_delta,
            window_size                = config.trend_window,
            noise_mult                 = config.noise_mult,
            min_steps                  = config.min_steps_before_stopping,
            plateau_threshold          = config.plateau_threshold,
            logit_std_delta            = config.logit_std_min_delta,
            phase3_patience_multiplier = config.phase3_patience_multiplier,
            eval_every                 = config.eval_every,
        )

    @staticmethod
    def _get_substrate(model: nn.Module) -> LorentzSubstrateV2:
        """Extract substrate from SafeHyperbolicModel or HyperbolicTransformerV2."""
        if hasattr(model, "substrate"):
            return model.substrate
        if hasattr(model, "core_model") and hasattr(model.core_model, "substrate"):
            return model.core_model.substrate
        raise AttributeError(
            "DistillationTrainerV2: student has no .substrate attribute. "
            "Use SafeHyperbolicModel or HyperbolicTransformerV2."
        )

    # ── checkpoint ────────────────────────────────────────────────────────

    def save(self, is_best: bool = False) -> None:
        ckpt = {
            "step":           self.step,
            "model":          self.student.state_dict(),
            "opt":            self.optimizer.state_dict(),
            "sched":          self.scheduler.state_dict(),
            "best_val":       self.best_val,
            "patience":       self.patience,
            "early_stopper":  self.early_stopper.state_dict(),
            "train_hist":     self.train_hist[-1000:],
            "val_hist":       self.val_hist,
            "config":         asdict(self.config),
            # Diagnostic EMA state (prevents reset artifacts on resume)
            "rdc_ema":        self._rdc_ema,
            "rdc_beta":       self._rdc_beta,
        }
        path = self.checkpoint_dir / f"distill_v2_ckpt_{self.step}.pt"
        torch.save(ckpt, path)
        self._ckpts.append(path)
        torch.save(ckpt, self.checkpoint_dir / "distill_v2_latest.pt")
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "distill_v2_best.pt")
            print("💾 Best v2 model saved!")
        while len(self._ckpts) > self.config.keep_last_n_checkpoints:
            old = self._ckpts.pop(0)
            if old.exists():
                old.unlink()

    def load_checkpoint(self, path) -> None:
        """Resume training from a checkpoint file (Path or str).
        Restores model weights, optimizer, scheduler, step counter,
        best_val, patience, and history. Safe to call before train().
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"🔄 Loading checkpoint: {path}")
        self.load(str(path))
        print(f"   ✅ Resumed at step={self.step}  best_val={self.best_val:.4f}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["opt"])
        self.scheduler.load_state_dict(ckpt["sched"])
        self.step      = ckpt.get("step",      0)
        self.best_val  = ckpt.get("best_val",  float("inf"))
        # Restore diagnostic EMA state to avoid reset artifacts on resume
        self._rdc_ema  = ckpt.get("rdc_ema",  0.0)
        self._rdc_beta = ckpt.get("rdc_beta",  0.95)
        self.patience  = ckpt.get("patience",  0)
        self.train_hist= ckpt.get("train_hist",[])
        self.val_hist  = ckpt.get("val_hist",  [])
        if "early_stopper" in ckpt:
            self.early_stopper.load_state_dict(ckpt["early_stopper"])
        print(f"✅ Loaded v2 checkpoint from step {self.step}")

    # ── single train step ─────────────────────────────────────────────────

    def distillation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.student.train()
        input_ids = batch["input_ids"].to(self.device)
        labels    = batch["labels"].to(self.device)

        # ── Bonus: Adaptive lambda_hidden warm-up ─────────────────────────
        # In early training, geometric alignment is noisy (model not converged).
        # Ramp lambda_hidden from 0 → target over warmup_steps to avoid
        # early gradient interference from a poorly-initialized projector.
        #
        # FIX: lambda_radius is NOT warmed up — it must be active from step 0.
        # The original code warmed up ALL geometric losses including radius,
        # leaving the radius loss at 0 during warmup. Without radius pressure,
        # the KL+CE objective alone drives embeddings to the origin (zero norm
        # lambda_hidden is warmed up: it depends on the teacher_proj which needs
        # a few steps to initialize. lambda_contrast and lambda_radius must be
        # active from step 0 — they are discrimination signals, not alignment signals.
        # Zeroing them during warmup causes the same collapse seen with lambda_radius.
        if self.config.adaptive_lambda and self.config.warmup_steps > 0:
            warmup_frac = min(1.0, self.step / self.config.warmup_steps)
            self.loss_fn.lambda_hidden   = self.config.lambda_hidden   * warmup_frac
            # lambda_contrast and lambda_radius: always full strength from step 0
            self.loss_fn.lambda_contrast = self.config.lambda_contrast
            self.loss_fn.lambda_radius   = self.config.lambda_radius

        # Teacher
        teacher_logits = teacher_hidden = None
        if self.teacher is not None and self.config.lambda_distill > 0:
            with torch.no_grad():
                t = self.teacher(input_ids, return_hidden=True)
                teacher_logits = t["logits"]
                teacher_hidden = t.get("hidden_states")

        # Student
        s = self.student(input_ids, labels=labels)
        student_logits = s["logits"]
        student_hidden = s.get("hidden_states")
        lm_loss        = s["loss"]

        # Distillation loss (includes geometric terms)
        distill_loss = torch.tensor(0.0, device=self.device)
        l_hidden = l_radius = l_contrast = torch.tensor(0.0, device=self.device)
        if teacher_logits is not None:
            d = self.loss_fn(
                student_logits = student_logits,
                teacher_logits = teacher_logits,
                labels         = labels,
                student_hidden = student_hidden,
                teacher_hidden = teacher_hidden,
            )
            distill_loss = d["total"]
            l_hidden     = d["l_hidden"]
            l_radius     = d["l_radius"]
            l_contrast   = d["l_contrast"]

        total = (
            (1.0 - self.config.lambda_distill) * lm_loss
            + self.config.lambda_distill * distill_loss
        )

        self.optimizer.zero_grad()
        total.backward()

        # ── Curvature-aware Riemannian gradient scaling (TRAINER BUG FIX) ─────
        # Standard Euclidean gradients are incorrect for manifold parameters because
        # they ignore the Riemannian metric tensor. For Lorentz manifold points x,
        # the Euclidean gradient g must be projected to the tangent space T_xH^n:
        #
        #   g_riem = g + ⟨g, x⟩_M · x    (Riemannian gradient on hyperboloid)
        #
        # where ⟨·,·⟩_M is the Minkowski inner product.
        # Additionally we scale by 1/K (curvature) to normalise across dims.
        #
        # We apply this only to parameters with ambient_dim shape (n+1),
        # i.e. embedding weights and LM head weights that live on H^n.
        # All other parameters (LayerNorm, attention projections) get
        # standard Euclidean gradient clipping.
        try:
            substrate = self.loss_fn.substrate
            K_val = substrate.K.item()
            ambient = substrate.n + 1
            # Collect parameters with gradients
            tangent_grads = [
                p for p in self.student.parameters()
                if p.grad is not None and p.dim() >= 1 and p.shape[-1] == ambient
            ]
            for p in tangent_grads:
                g = p.grad.data
                x = p.data
                # Minkowski inner product ⟨g, x⟩_M = -g₀x₀ + Σgᵢxᵢ
                # Reshape to [..., ambient] for broadcasting
                g_flat = g.reshape(-1, ambient)
                x_flat = x.reshape(-1, ambient)
                inner = (
                    (g_flat[:, 1:] * x_flat[:, 1:]).sum(dim=-1)
                    - g_flat[:, 0] * x_flat[:, 0]
                )   # [N]
                # Project: g_riem = g + inner * x
                correction = (inner.unsqueeze(-1) * x_flat).reshape_as(g)
                g.add_(correction)
                # Scale by 1/max(K, 1e-4) to normalise curvature effect
                g.mul_(1.0 / max(K_val, 1e-4))
        except Exception:
            pass  # Fall back to standard clipping if substrate unavailable

        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.gradient_clip
        )
        # Also clip teacher_proj gradients (separate param group)
        if self.loss_fn.teacher_proj is not None:
            torch.nn.utils.clip_grad_norm_(
                self.loss_fn.teacher_proj.parameters(), self.config.gradient_clip
            )
        # ── Riemannian natural gradient correction (Information Geometry) ──────
        #
        # On H^n the metric tensor g_ij scales as (sinh(r)/r)² at geodesic
        # radius r. Standard Adam ignores this and overestimates the step size.
        # The diagonal natural gradient correction (Amari 1998, Bonnabel 2013):
        #
        #   g_corrected = g_euclidean * (r / sinh(r))
        #
        # Controls which parameter groups receive correction via config flags:
        #
        #   riemannian_correct_vocab   : lm_head tangent weights [V, n]
        #   riemannian_correct_embed   : token embedding weights
        #   riemannian_correct_encoder : all encoder parameters on manifold
        #
        # Flags default to vocab-only (validated). Others can be enabled
        # incrementally as evidence accumulates. Set in CFG['training'].
        _corr_vocab   = getattr(self.config, 'riemannian_correct_vocab',   True)
        _corr_embed   = getattr(self.config, 'riemannian_correct_embed',   False)
        _corr_encoder = getattr(self.config, 'riemannian_correct_encoder', False)

        if _corr_vocab or _corr_embed or _corr_encoder:
            try:
                n_ambient = self.loss_fn.substrate.n + 1
                n_tangent = self.loss_fn.substrate.n

                for name, param in self.student.named_parameters():
                    if param.grad is None:
                        continue

                    # Classify parameter
                    is_vocab   = 'lm_head' in name and 'weight' in name
                    is_embed   = 'embed' in name and 'weight' in name
                    is_encoder = (not is_vocab and not is_embed
                                  and param.dim() >= 2)

                    # Check if correction is enabled for this group
                    if is_vocab   and not _corr_vocab:   continue
                    if is_embed   and not _corr_embed:   continue
                    if is_encoder and not _corr_encoder: continue
                    if not (is_vocab or is_embed or is_encoder): continue

                    w = param.data
                    # Determine radius based on parameter shape
                    if w.shape[-1] == n_tangent:
                        # Tangent-space storage [*, n]: use L2 norm directly
                        r = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    elif w.shape[-1] == n_ambient:
                        # Ambient-space storage [*, n+1]: radius from spatial components
                        r = w[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    else:
                        # Non-manifold parameter (e.g. LayerNorm, bias): skip
                        continue

                    # Natural gradient correction: r/sinh(r) ∈ (0, 1]
                    # Approaches 1 near origin (no effect), shrinks at high radius
                    correction = r / torch.sinh(r.clamp(max=10.0))
                    param.grad.data.mul_(correction)

            except Exception:
                pass  # graceful fallback to standard Euclidean step

        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        # Keep loss_fn informed of current step for collapse warning throttle
        self.loss_fn._global_step   = self.step
        self.loss_fn._warmup_steps  = self.config.warmup_steps

        # ── Diversity metric (correct: multinomial sampling, not argmax) ───
        with torch.no_grad():
            logit_std  = student_logits.float().std().item()
            seq_logits = student_logits[0].float()
            probs      = F.softmax(seq_logits, dim=-1)
            sampled    = torch.multinomial(probs, num_samples=1).squeeze(-1)
            unique_ratio = len(sampled.unique()) / max(sampled.numel(), 1)

            # Update RDC EMA using current logit_std and l_hidden
            _l_hid_val = l_hidden.item() if isinstance(l_hidden, torch.Tensor) else l_hidden
            _rdc_raw   = logit_std / (_l_hid_val + 1e-2)
            self._rdc_ema = self._rdc_beta * self._rdc_ema + (1.0 - self._rdc_beta) * _rdc_raw

            # FIX: compute actual mean radius for logging (not the loss value)
            # l_radius is now a loss scalar (mix of -mean + MSE), not the radius itself.
            # Log the true mean geodesic radius separately so telemetry is interpretable.
            mean_radius = 0.0
            w_std       = 0.0   # contrastive weight std — tracks pooling health
            w_entropy   = 0.0   # contrastive weight entropy — tracks pooling diversity
            if student_hidden is not None:
                sh = student_hidden
                ambient = self.loss_fn.substrate.n + 1
                if sh.shape[-1] == ambient:
                    K    = self.loss_fn.substrate.K.to(sh.device, sh.dtype)
                    sqK  = torch.sqrt(K)
                    x0   = sh[..., 0]
                    arg  = (sqK * x0).clamp(min=1.0 + 1e-6)
                    rad  = torch.log(arg + torch.sqrt(arg**2 - 1.0 + 1e-15)) / sqK
                    mean_radius = rad.mean().item()

                    # Norm-collapse diagnostics (informational — not used for pooling).
                    # Pooling is now last-token (immune to norm variation), so these
                    # metrics serve as architectural health checks:
                    #
                    # Expected state (correct, not a bug):
                    #   w_std ≈ 0      → LayerNorm forces uniform tangent norms
                    #   w_entropy ≈ log(L) = 4.85 → uniform weights, maximum entropy
                    #   This confirms RiemannianLayerNorm is working as designed.
                    #
                    # Unexpected states (investigate if seen):
                    #   w_entropy << log(L): one position has much larger norm than others
                    #   w_std > 0.05: norms are non-uniform (LayerNorm may be bypassed)
                    #
                    # Interpretation table (with last-token pooling):
                    #   w_std  | w_entropy  | diagnosis
                    #   -------|------------|--------------------------------------
                    #   ≈0     | ≈log(L)    | EXPECTED — LayerNorm healthy ✅
                    #   ≈0     | << log(L)  | norm spike — one position dominating ⚠️
                    #   >0.05  | any        | LayerNorm bypassed or broken ⚠️
                    s_spatial_log = self.loss_fn._student_to_tangent(sh)  # [B, L, n]
                    eps = 1e-3
                    norms = s_spatial_log.norm(dim=-1, keepdim=True).clamp(min=0.0)
                    w = (norms + eps) / (norms + eps).sum(dim=1, keepdim=True)
                    w_std     = w.std().item()
                    w_entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean().item()

        return {
            "step":         self.step,
            "loss":         total.item(),
            "lm_loss":      lm_loss.item(),
            "distill_loss": distill_loss.item(),
            "l_hidden":     l_hidden.item() if isinstance(l_hidden, torch.Tensor) else l_hidden,
            "l_radius":     l_radius.item() if isinstance(l_radius, torch.Tensor) else l_radius,
            "l_contrast":   l_contrast.item() if isinstance(l_contrast, torch.Tensor) else l_contrast,
            "mean_radius":  mean_radius,   # FIX: true radius mean for telemetry
            "w_std":        w_std,         # contrastive weight std  (>0.01 = healthy pooling)
            "w_entropy":    w_entropy,     # contrastive weight entropy (high = diverse positions)
            "ppl":          math.exp(min(lm_loss.item(), 20.0)),
            "lr":           self.scheduler.get_last_lr()[0],
            "logit_std":    logit_std,
            "diversity":    unique_ratio,
            # ── Radial Drift Coefficient (proxy) ────────────────────────────
            # RDC_proxy = logit_std / (l_hidden + ε)
            #
            # Theoretical basis (implicit margin maximization in Lorentz space):
            #   - logit_std ∝ sinh²(r) × logit_scale  → proxy for radial scale
            #   - l_hidden = 1 - cos(student, teacher)  → angular misalignment
            #   - When l_hidden → 0: angular convergence complete
            #   - When logit_std remains high: radial scale still dominates
            #   - RDC_proxy → ∞ signals "Degenerate Equilibrium":
            #       ∇_ang → 0 (semantics locked)
            #       radial scale high but STABLE (not runaway)
            #       system at confidence equilibrium, not functional minimum
            #
            # Note: true RDC requires explicit ∇_rad/∇_ang decomposition (expensive).
            # This proxy captures the same signal cheaply from observables.
            # The EMA version (rdc_ema) smooths per-step noise.
            #
            # Interpretation:
            #   rdc_proxy  < 5 : angular learning dominates ✅ (Phase 1)
            #   rdc_proxy 5-20 : transition (both active)   ⚠️ (Phase 2)
            #   rdc_proxy > 20 : Degenerate Equilibrium     ❌ (Phase 3+)
            "rdc_proxy":    logit_std / ((l_hidden.item() if isinstance(l_hidden, torch.Tensor) else l_hidden) + 1e-2),
            "rdc_ema":      self._rdc_ema,   # smoothed version for regime detection
        }

    # ── eval ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.student.eval()
        total_lm = total_distill = total_tokens = 0.0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)

            s      = self.student(input_ids, labels=labels)
            lm_loss= s["loss"]
            if torch.isnan(lm_loss):
                continue

            total_lm     += lm_loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

            if self.teacher is not None and self.config.lambda_distill > 0:
                t = self.teacher(input_ids, return_hidden=False)
                # Pass labels correctly so CE is computed inside loss_fn
                d = self.loss_fn(
                    student_logits=s["logits"],
                    teacher_logits=t["logits"],
                    labels=labels,
                )
                total_distill += d["total"].item() * input_ids.numel()

        n = max(total_tokens, 1.0)
        lm = total_lm / n
        return {
            "val_loss":    lm,
            "val_distill": total_distill / n,
            "val_ppl":     math.exp(min(lm, 20.0)),
        }

    # ── train loop ────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple:
        print("\n" + "=" * 60)
        print("🎓 STARTING v2 TRAINING")
        print("=" * 60)
        print(f"Teacher:        {'ON' if self.teacher else 'OFF'}")
        print(f"Lambda distill: {self.config.lambda_distill}")
        print(f"Max steps:      {self.config.max_steps}")
        print("=" * 60)

        data_iter = iter(train_loader)

        while self.step < self.config.max_steps and not self.stop:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            m = self.distillation_step(batch)
            self.train_hist.append(m)

            # Feed loss values to LossBalancer EMA tracker
            self.loss_balancer.update_ema({
                "l_hidden":   m.get("l_hidden",   0.0),
                "l_contrast": m.get("l_contrast", 0.0),
                "l_soft":     m.get("distill_loss", 0.0),
            })

            if self.step % self.config.log_every == 0:
                logit_std = m.get('logit_std', 0.0)
                diversity = m.get('diversity', 0.0)
                std_flag = "✅" if logit_std > 0.5 else ("⚠️" if logit_std > 0.1 else "❌")
                div_flag = "✅" if diversity > 0.3 else ("⚠️" if diversity > 0.1 else "❌")
                l_hid  = m.get('l_hidden',   0.0)
                l_rad  = m.get('l_radius',   0.0)
                l_con  = m.get('l_contrast', 0.0)
                # FIX: log mean_radius (true geodesic distance) not l_radius (loss value)
                # l_radius is now a mix of -mean + MSE, uninterpretable as "rad" metric
                mean_rad = m.get('mean_radius', 0.0)
                w_std_v  = m.get('w_std', 0.0)
                w_ent_v  = m.get('w_entropy', 0.0)
                rdc_v    = m.get('rdc_proxy', 0.0)
                # w_entropy ≈ log(128)=4.85 is EXPECTED and correct (LayerNorm uniform norms).
                # Flag ❌ only if entropy << log(L) (norm spike) or w_std > 0.05 (LN broken).
                rad_flag = "✅" if mean_rad > 0.5 else ("⚠️" if mean_rad > 0.1 else "❌")
                w_flag   = "✅" if w_std_v  < 0.05 else ("⚠️" if w_std_v < 0.1 else "❌")
                h_flag   = "✅" if w_ent_v  > 4.0  else ("⚠️" if w_ent_v > 2.0  else "❌")
                # RDC proxy: logit_std / (l_hidden + ε) — radial vs angular signal ratio
                # <5 = angular learning dominates ✅ | 5-20 = transition ⚠️ | >20 = radial drift ❌
                rdc_flag = "✅" if rdc_v < 5 else ("⚠️" if rdc_v < 20 else "❌")
                print(
                    f"  step={self.step:>6}  "
                    f"loss={m['loss']:.4f}  "
                    f"ppl={m['ppl']:.1f}  "
                    f"lr={m['lr']:.2e}  "
                    f"logit_std={logit_std:.4f}{std_flag}  "
                    f"div={diversity:.2f}{div_flag}  "
                    f"hid={l_hid:.3f}  rad={mean_rad:.3f}{rad_flag}  "
                    f"ctr={l_con:.3f}  rdc={rdc_v:.1f}{rdc_flag}  "
                    f"w_std={w_std_v:.4f}{w_flag}  w_ent={w_ent_v:.2f}{h_flag}"
                )

            if self.step % self.config.eval_every == 0:
                vm = self.evaluate(val_loader)
                vm["step"] = self.step  # FIX: inject step before appending

                # ── EarlyStoppingV3 ───────────────────────────────────────
                last_logit_std = (
                    self.train_hist[-1].get("logit_std", 0.0)
                    if self.train_hist else 0.0
                )
                should_stop, info = self.early_stopper.step(
                    val_loss     = vm["val_loss"],
                    logit_std    = last_logit_std,
                    current_step = self.step,
                )

                # FIX: enrich vm with early-stopper signals for full CSV schema
                vm["ema_slow"] = info["ema"]
                vm["ema_fast"] = info["ema_fast"]
                vm["noise"]    = info["noise"]
                vm["phase"]    = info["phase"]
                vm["sg"]       = info["steps_since_global"]

                self.val_hist.append(vm)

                # ── Regime health check ───────────────────────────────────
                if self.step >= self.config.warmup_steps:
                    recent = self.train_hist[-self.config.log_every:] if self.train_hist else []
                    r_logit_std = sum(m.get("logit_std", 0) for m in recent) / max(len(recent), 1)
                    r_hid       = sum(m.get("l_hidden",  0) for m in recent) / max(len(recent), 1)
                    r_ctr       = sum(m.get("l_contrast",0) for m in recent) / max(len(recent), 1)

                    issues = []
                    hints  = []
                    codes  = []   # structured codes for groupby analysis

                    if r_logit_std < 1.0:
                        issues.append(f"logit_std={r_logit_std:.3f} < 1.0 → UNDERCONFIDENT")
                        hints.append("try logit_scale clamp ↑ or temperature ↑ 1.2")
                        codes.append("UNDERCONFIDENT")
                    if r_logit_std > 2.5:
                        issues.append(f"logit_std={r_logit_std:.3f} > 2.5 → OVERCONFIDENT")
                        hints.append("consider logit_scale clamp ↓ 4.0")
                        codes.append("OVERCONFIDENT")
                    if r_hid > 0.5 and self.step > 500:
                        issues.append(f"hid={r_hid:.3f} > 0.5 → KL DOMINANCE")
                        hints.append("try temperature ↑ 1.2, lambda_distill ↓ 0.30")
                        codes.append("KL_DOMINANCE")
                    if r_ctr < 0.05 and self.step < 3000:
                        issues.append(f"ctr={r_ctr:.3f} < 0.05 before step 3000 → COLLAPSE")
                        hints.append("monitor for 200 more steps before acting")
                        codes.append("CTR_COLLAPSE")

                    # Degenerate Equilibrium detection (RDC-based).
                    # Fires when angular convergence is complete (l_hidden low and stable)
                    # while radial scale is high (rdc_ema >> threshold).
                    # This is the correct framing: NOT "runaway growth" but a stable
                    # equilibrium dominated by confidence scaling, not semantic learning.
                    # Implication: further training optimizes scale, not representations.
                    # The OBJECTIVE dynamics still favor radial even though the manifold
                    # constraint limits the EMBEDDINGS — radial drift has migrated to
                    # vocabulary embedding norms (Euclidean, unconstrained by manifold ops).
                    # Action: no tuner adjustment (this is not a crisis — it's an equilibrium).
                    # It's logged as DEGENERATE_EQ for post-hoc analysis and paper evidence.
                    if (self.step >= self.config.warmup_steps
                            and self._rdc_ema > 15
                            and r_hid < 0.25
                            and r_logit_std > 2.0):
                        issues.append(
                            f"rdc_ema={self._rdc_ema:.1f} > 15  hid={r_hid:.3f} < 0.25"
                            f"  → DEGENERATE EQUILIBRIUM (angular done, scale dominates)"
                        )
                        hints.append(
                            "Informational only: system at confidence equilibrium. "
                            "Scale-invariant objective or vocab embedding regularization "
                            "would resolve this structurally."
                        )
                        codes.append("DEGENERATE_EQ")

                    # PPL stagnation: 2-eval window to filter noise.
                    if len(self.val_hist) >= 3:
                        ppl_cur  = self.val_hist[-1]["val_ppl"]
                        ppl_2ago = self.val_hist[-3]["val_ppl"]
                        ppl_1ago = self.val_hist[-2]["val_ppl"]
                        delta_recent = (ppl_1ago  - ppl_cur)  / max(ppl_1ago,  1)
                        delta_prev   = (ppl_2ago  - ppl_1ago) / max(ppl_2ago,  1)
                        if delta_recent < 0.01 and delta_prev < 0.01 and self.step < 2000:
                            issues.append(
                                f"ppl stagnant 2 evals "
                                f"({ppl_2ago:.0f}→{ppl_1ago:.0f}→{ppl_cur:.0f}, "
                                f"Δ={delta_recent*100:.2f}%)"
                            )
                            hints.append("try temperature ↑ 1.2, lambda_distill ↓ 0.33")
                            codes.append("PPL_STAGNANT")

                    regime_status = "ALERT" if issues else "OK"
                    regime_reason = " | ".join(issues) if issues else ""
                    regime_code   = "|".join(codes) if codes else "OK"

                    # Attach to val_hist for CSV export and post-hoc analysis
                    self.val_hist[-1]["regime_status"] = regime_status
                    self.val_hist[-1]["regime_reason"] = regime_reason
                    self.val_hist[-1]["regime_code"]   = regime_code
                    self.val_hist[-1]["rdc_ema"]       = round(self._rdc_ema, 3)
                    # "regime" = simplified single label for groupby analysis
                    if "DEGENERATE_EQ" in regime_code:
                        self.val_hist[-1]["regime"] = "DEGENERATE_EQ"
                    elif regime_status == "OK":
                        self.val_hist[-1]["regime"] = "STABLE"
                    else:
                        # First non-DEQ code, or ALERT
                        first_code = next((c for c in codes if c != "DEGENERATE_EQ"), regime_code)
                        self.val_hist[-1]["regime"] = first_code if first_code else "ALERT"

                    if issues:
                        print("  ⚠️  REGIME ALERT:")
                        for iss in issues:
                            flag = "ℹ️" if "DEGENERATE" in iss else "❌"
                            print(f"      {flag} {iss}")
                        for h in hints:
                            print(f"      🔧 {h}")
                    else:
                        print(f"  ✅ Regime OK  "
                              f"logit_std={r_logit_std:.2f}  "
                              f"hid={r_hid:.3f}  "
                              f"ctr={r_ctr:.3f}  "
                              f"rdc_ema={self._rdc_ema:.1f}")

                    # ── DegEqController — post-convergence intervention ───
                    deq_applied = self.deg_eq_ctrl.step(
                        regime_code = regime_code,
                        step        = self.step,
                        val_hist    = self.val_hist,
                        optimizer   = self.optimizer,
                    )
                    if deq_applied:
                        for rec in deq_applied:
                            if rec.get('action') == 'stop':
                                self.stop = True

                    # ── Adaptive tuner — regime-driven crisis response ────
                    tuner_applied = self.adaptive_tuner.step(regime_code, self.step, self.val_hist)

                    # AdaptiveTuner priority: freeze params it just touched in
                    # LossBalancer for the next cooldown window to prevent conflict.
                    # E.g.: CTR_COLLAPSE triggers λ_contrast ↓; LossBalancer must
                    # not immediately push it back ↑.
                    if tuner_applied:
                        self.loss_balancer.skip_params = {
                            rec["param"] for rec in tuner_applied
                        }
                    elif regime_code == "OK":
                        self.loss_balancer.skip_params = set()   # unfreeze on clean eval

                    # ── Loss balancer — proactive EMA-based equalization ──
                    # Only runs when no active crisis (skip_params may gate it).
                    balance_applied = self.loss_balancer.balance(self.step)
                    if balance_applied:
                        for rec in balance_applied:
                            _sig = rec.get("signal", "ema_loss")
                            print(
                                f"  ⚖️  LossBalance [{_sig}]  {rec['param']}: "
                                f"{rec['before']:.4f} → {rec['after']:.4f}  "
                                f"(eff {rec['eff_before']:.4f} → target {rec['target']:.4f}"
                                + (f"  meta_var={rec['meta_var']:.5f}" if 'meta_var' in rec else "")
                                + ")"
                            )
                        if self.val_hist:
                            self.val_hist[-1]["balance_log"] = balance_applied
                # ─────────────────────────────────────────────────────────

                phase_str = f"P{info['phase']}"
                gate_str  = info["noise_gate"]
                reason    = info["decision"] or ""
                print(
                    f"  📊 Val  step={self.step}  "
                    f"loss={vm['val_loss']:.4f}  "
                    f"ppl={vm['val_ppl']:.1f}  "
                    f"ema={info['ema']:.4f}  "
                    f"ef={info['ema_fast']:.4f}  "
                    f"noise={info['noise']:.4f}  "
                    f"P{info['phase']}  "
                    f"pat={info['patience']:>2}/{info['effective_patience']}  "
                    f"sg={info['steps_since_global']}"
                    + (f"  [{info['noise_gate']}]"
                       if info["noise_gate"] not in ("is_best", "") else "")
                    + (f"  [{info['decision']}]"
                       if info["decision"] and not should_stop else "")
                )

                if info["is_best"]:
                    self.best_val = vm["val_loss"]
                    self.patience = 0
                    self.save(is_best=True)
                else:
                    self.patience = info["patience"]

                if should_stop:
                    print(
                        f"  ⏹  Early stopping at step {self.step}\n"
                        f"     {info['decision']}"
                    )
                    self.stop = True

            if self.step % self.config.checkpoint_every == 0:
                self.save()
                # ── GradNorm audit — real per-loss gradient contribution ──
                # Probe: final LayerNorm weight (128 params, 0.6% overhead).
                # Requires loss tensors from the LAST distillation step.
                # We re-use the saved loss_fn state — no extra forward pass needed
                # for the EMA-based report; full probe measurement is an option
                # when loss tensors are retained (requires retain_graph in step).
                try:
                    _ema = self.loss_balancer._ema
                    _lh  = getattr(self.loss_fn, 'lambda_hidden',   0.0)
                    _lc  = getattr(self.loss_fn, 'lambda_contrast',  0.0)
                    _ld  = self.config.lambda_distill * 0.5

                    # Check if real grad norms available (from GradNormMeasure)
                    _gnorm_kl  = _ema.get('gnorm_l_soft',     None)
                    _gnorm_hid = _ema.get('gnorm_l_hidden',   None)
                    _gnorm_ctr = _ema.get('gnorm_l_contrast', None)

                    if _gnorm_kl is not None:
                        # Real grad norms: ground truth
                        _effs = {"kl": _gnorm_kl, "hidden": _gnorm_hid or 0, "contrast": _gnorm_ctr or 0}
                        _source = "REAL"
                    else:
                        # EMA proxy fallback
                        _effs = {
                            "kl":      _ld * _ema.get('l_soft',     0),
                            "hidden":  _lh * _ema.get('l_hidden',   0),
                            "contrast":_lc * _ema.get('l_contrast', 0),
                        }
                        _source = "EMA"

                    _active = {k: v for k, v in _effs.items() if v > 1e-10}
                    _imb = (max(_active.values()) / min(_active.values())
                            if len(_active) >= 2 else 1.0)
                    _meta_var = (sum((v - sum(_active.values())/len(_active))**2
                                     for v in _active.values()) / len(_active)
                                  if _active else 0.0)

                    _imb_flag = "✅" if _imb < 3 else ("⚠️" if _imb < 8 else "❌")
                    print(
                        f"  📐 GradBalance [{_source}]  step={self.step}  "
                        + "  ".join(f"eff_{k}={v:.4f}" for k, v in _effs.items())
                        + f"  imbalance={_imb:.1f}x{_imb_flag}"
                        + f"  meta_var={_meta_var:.5f}"
                    )

                    # KL dominance (imbalance 20-40x) is EXPECTED in distillation:
                    # the teacher signal inherently has larger gradient magnitude.
                    # Only flag structural imbalance (> 50x) which indicates bounds
                    # preventing the LossBalancer from doing its job.
                    if _imb > 50 and self.step > self.config.warmup_steps + 400:
                        import warnings
                        warnings.warn(
                            f"[GradImbalance] imbalance={_imb:.1f}x > 8x at step {self.step}. "
                            f"LossBalancer bounds may be too tight. "
                            f"Consider widening BOUNDS in LossBalancer.",
                            RuntimeWarning
                        )

                    self.grad_norm_measure.history.append({
                        "step": self.step, "source": _source,
                        **_effs, "imbalance_ratio": round(_imb, 2),
                        "meta_var": round(_meta_var, 6),
                    })
                except Exception:
                    pass

                # ── Incremental CSV flush ─────────────────────────────────
                _flush = getattr(self, '_flush_csv', None)
                if _flush is not None:
                    try:
                        _flush(self.train_hist, self.val_hist)
                    except Exception as _e:
                        import warnings
                        warnings.warn(f"[IncrementalCSV] flush failed: {_e}", RuntimeWarning)

        print("=" * 60)
        print("✅ Training complete.")
        return self.train_hist, self.val_hist
