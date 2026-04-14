"""Custom loss functions with gradient and Hessian for gradient boosting.

The core problem: CatBoost and LightGBM use second-order Newton-Raphson
optimization.  The standard Quantile (Pinball) loss is piecewise-linear,
so its Hessian is zero everywhere — collapsing Newton steps to unscaled
gradient descent, which converges extremely slowly.

This module provides two superior alternatives:

1. **Expectile loss** — The quadratic asymmetric counterpart of Quantile.
   Fully differentiable, non-zero Hessian, fast Newton convergence.
   Controls asymmetry via τ ∈ (0, 1), similar to quantile level.

2. **LINEX loss** — Linear-Exponential.  Underestimation is penalized
   exponentially, overestimation linearly.  The gold standard for extreme
   asymmetric cost scenarios (SLA penalties, OOM prevention).

Both provide:
- Pure NumPy vectorized gradient/hessian (always available)
- Optional Numba JIT-compiled versions (when numba is installed)
- CatBoost-compatible ``calc_ders_range`` interface
- LightGBM-compatible ``(grad, hess)`` tuple interface

References:
    - Toth (2016), Laimighofer (2022): Expectile predictive superiority
    - Shrivastava et al. (2023): LINEX geometric robustness
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Try to import numba for JIT; graceful fallback.
try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ────────────────────── Expectile Loss ──────────────────────


def expectile_grad_hess(
    y: np.ndarray, f: np.ndarray, tau: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized gradient and Hessian for Expectile loss.

    L(y, f) = |τ - I(y < f)| · (y - f)²

    Gradient:  g = -2 · w · (y - f)
    Hessian:   h =  2 · w

    where w = τ if y ≥ f, else (1 - τ).

    Args:
        y: True values.
        f: Predicted values.
        tau: Asymmetry parameter in (0, 1).  tau > 0.5 penalizes
             underestimation more heavily.

    Returns:
        (gradient, hessian) arrays, same shape as y.
    """
    residual = y - f
    w = np.where(residual >= 0, tau, 1.0 - tau)
    grad = -2.0 * w * residual
    hess = 2.0 * w
    return grad, hess


def expectile_loss(y: np.ndarray, f: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Compute per-sample Expectile loss values."""
    residual = y - f
    w = np.where(residual >= 0, tau, 1.0 - tau)
    return w * residual ** 2


# ────────────────────── LINEX Loss ──────────────────────


def linex_grad_hess(
    y: np.ndarray, f: np.ndarray, a: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized gradient and Hessian for LINEX loss.

    L(y, f) = exp(a·(y - f)) - a·(y - f) - 1

    Gradient:  g = -(exp(a·z) - 1)   where z = y - f, then w.r.t. f:
               g = a·exp(a·z) - a  →  simplified: a·(exp(a·z) - 1)
               (sign depends on derivative w.r.t. f, not y)
    Hessian:   h = a² · exp(a·z)

    When a > 0: underestimation (z > 0) is penalized exponentially.
    When a < 0: overestimation is penalized exponentially.

    For SLA protection, use a > 0 (penalize under-provisioning).

    Args:
        y: True values.
        f: Predicted values.
        a: Asymmetry parameter.  Magnitude controls severity;
           sign controls direction.  Typical: 0.5–3.0.

    Returns:
        (gradient, hessian) arrays, same shape as y.
    """
    z = y - f
    # Clamp to prevent overflow in exp
    az = np.clip(a * z, -50.0, 50.0)
    exp_az = np.exp(az)

    # Derivative of L w.r.t. f (note: dL/df = -dL/dz)
    grad = -a * (exp_az - 1.0)
    hess = a * a * exp_az
    return grad, hess


def linex_loss(y: np.ndarray, f: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Compute per-sample LINEX loss values."""
    z = y - f
    az = np.clip(a * z, -50.0, 50.0)
    return np.exp(az) - a * z - 1.0


# ────────────────────── Quantile (Pinball) Loss ──────────────────────


def quantile_grad_hess(
    y: np.ndarray, f: np.ndarray, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Standard quantile (pinball) gradient and Hessian.

    Included for completeness/comparison.  Note: Hessian is forced to 1.0
    (the standard hack used by boosting frameworks), which degrades to
    unscaled gradient descent.

    Args:
        y: True values.
        f: Predicted values.
        alpha: Quantile level in (0, 1).

    Returns:
        (gradient, hessian) arrays.
    """
    residual = y - f
    grad = np.where(residual >= 0, -alpha, 1.0 - alpha)
    hess = np.ones_like(residual)  # Forced constant — Newton collapses
    return grad, hess


# ────────────────────── Numba JIT Versions ──────────────────────


def get_numba_expectile(tau: float = 0.5) -> Callable | None:
    """Return Numba JIT-compiled Expectile gradient/hessian, or None."""
    if not HAS_NUMBA:
        return None

    @nb.njit(cache=True)
    def _calc(y, f):  # pragma: no cover
        n = len(y)
        grad = np.empty(n)
        hess = np.empty(n)
        for i in range(n):
            r = y[i] - f[i]
            w = tau if r >= 0.0 else (1.0 - tau)
            grad[i] = -2.0 * w * r
            hess[i] = 2.0 * w
        return grad, hess

    return _calc


def get_numba_linex(a: float = 1.0) -> Callable | None:
    """Return Numba JIT-compiled LINEX gradient/hessian, or None."""
    if not HAS_NUMBA:
        return None

    @nb.njit(cache=True)
    def _calc(y, f):  # pragma: no cover
        n = len(y)
        grad = np.empty(n)
        hess = np.empty(n)
        for i in range(n):
            z = y[i] - f[i]
            az = a * z
            if az > 50.0:
                az = 50.0
            elif az < -50.0:
                az = -50.0
            e = np.exp(az)
            grad[i] = -a * (e - 1.0)
            hess[i] = a * a * e
        return grad, hess

    return _calc


# ────────────────────── CatBoost Interface ──────────────────────


class CatBoostExpectile:
    """CatBoost-compatible custom objective for Expectile regression.

    Usage:
        model = CatBoost({"loss_function": CatBoostExpectile(tau=0.8)})
    """

    def __init__(self, tau: float = 0.5):
        self.tau = tau

    def calc_ders_range(self, approxes, targets, weights):
        y = np.array(targets)
        f = np.array(approxes)
        g, h = expectile_grad_hess(y, f, self.tau)
        if weights is not None:
            w = np.array(weights)
            g *= w
            h *= w
        return list(zip(g.tolist(), h.tolist()))


class CatBoostLINEX:
    """CatBoost-compatible custom objective for LINEX regression.

    Usage:
        model = CatBoost({"loss_function": CatBoostLINEX(a=1.5)})
    """

    def __init__(self, a: float = 1.0):
        self.a = a

    def calc_ders_range(self, approxes, targets, weights):
        y = np.array(targets)
        f = np.array(approxes)
        g, h = linex_grad_hess(y, f, self.a)
        if weights is not None:
            w = np.array(weights)
            g *= w
            h *= w
        return list(zip(g.tolist(), h.tolist()))


# ────────────────────── LightGBM Interface ──────────────────────


def lgbm_expectile_objective(tau: float = 0.5):
    """Return a LightGBM-compatible objective function for Expectile."""

    def _objective(y, f):
        g, h = expectile_grad_hess(np.array(y), np.array(f.get_label()), tau)
        return g, h

    return _objective


def lgbm_linex_objective(a: float = 1.0):
    """Return a LightGBM-compatible objective function for LINEX."""

    def _objective(y, f):
        g, h = linex_grad_hess(np.array(y), np.array(f.get_label()), a)
        return g, h

    return _objective
