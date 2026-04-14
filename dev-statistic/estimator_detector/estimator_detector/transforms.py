"""Power transforms for target normalization — Box-Cox, Yeo-Johnson, and friends.

Gradient boosting models converge faster and produce tighter residuals when
the target variable is approximately symmetric.  This module provides:

1. **Box-Cox**: Classical power transform for strictly positive data.
   Finds λ that maximizes log-likelihood of normality.  O(N) per evaluation.
2. **Yeo-Johnson**: Generalization of Box-Cox that handles zero and negative
   values.  Preferred when data contains zeros (sparse/count distributions).
3. **Log1p**: log(1 + x) — fast, interpretable, good for right-skewed data.
4. **Sqrt**: Square root — mild variance stabilizer for count data.
5. **Auto-select**: Tries all applicable transforms, picks the one that
   minimizes |skewness| of the transformed data.

All transforms are O(N), well within the sub-second latency budget.
Inverse transforms are provided for back-transforming predictions.

Usage:
    from estimator_detector.transforms import auto_transform, TransformResult

    result = auto_transform(data)
    print(result.method)         # "box_cox"
    print(result.lambda_param)   # 0.23
    transformed = result.transformed
    original = result.inverse(transformed)

Integration with the detector:
    rec = detector.analyze(data)
    print(rec.transform)         # TransformResult if recommended

References:
    - Box & Cox (1964): Original power transform
    - Yeo & Johnson (2000): Extension to non-positive data
    - scipy.stats.boxcox, scipy.stats.yeojohnson
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import scipy.stats as sp_stats


class TransformMethod(Enum):
    """Available transform methods."""

    NONE = "none"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    LOG1P = "log1p"
    SQRT = "sqrt"


@dataclass
class TransformResult:
    """Result of a power transform recommendation/application.

    Attributes:
        method: Which transform was selected.
        transformed: The transformed array.
        lambda_param: The λ parameter (Box-Cox/Yeo-Johnson) or None.
        original_skew: Skewness before transform.
        transformed_skew: Skewness after transform.
        inverse_fn: Function to back-transform predictions.
        improvement: Reduction in |skewness| (0 to 1 scale).
    """

    method: TransformMethod
    transformed: np.ndarray
    lambda_param: float | None
    original_skew: float
    transformed_skew: float
    inverse_fn: Callable[[np.ndarray], np.ndarray]
    improvement: float

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Back-transform predictions to original scale."""
        return self.inverse_fn(np.asarray(values, dtype=float))

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "lambda_param": round(self.lambda_param, 6) if self.lambda_param is not None else None,
            "original_skew": round(self.original_skew, 4),
            "transformed_skew": round(self.transformed_skew, 4),
            "improvement": round(self.improvement, 4),
        }


# ────────────────────── Individual Transforms ──────────────────────


def box_cox(v: np.ndarray) -> TransformResult | None:
    """Apply Box-Cox transform.  Requires all values > 0.

    Box-Cox finds λ such that:
        y(λ) = (x^λ - 1) / λ   if λ ≠ 0
        y(λ) = log(x)           if λ = 0

    Args:
        v: 1D array of strictly positive values.

    Returns:
        TransformResult or None if data contains non-positive values.
    """
    if np.any(v <= 0):
        return None

    original_skew = float(sp_stats.skew(v))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            transformed, lam = sp_stats.boxcox(v)
            lam = float(lam)
        except Exception:
            return None

    transformed_skew = float(sp_stats.skew(transformed))
    improvement = _skew_improvement(original_skew, transformed_skew)

    def _inverse(y: np.ndarray) -> np.ndarray:
        return _boxcox_inverse(y, lam)

    return TransformResult(
        method=TransformMethod.BOX_COX,
        transformed=transformed,
        lambda_param=lam,
        original_skew=original_skew,
        transformed_skew=transformed_skew,
        inverse_fn=_inverse,
        improvement=improvement,
    )


def yeo_johnson(v: np.ndarray) -> TransformResult:
    """Apply Yeo-Johnson transform.  Handles zero and negative values.

    Generalization of Box-Cox that works for any real-valued data.

    Args:
        v: 1D array of numeric values.

    Returns:
        TransformResult.
    """
    original_skew = float(sp_stats.skew(v))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            transformed, lam = sp_stats.yeojohnson(v)
            lam = float(lam)
        except Exception:
            return TransformResult(
                method=TransformMethod.NONE,
                transformed=v.copy(),
                lambda_param=None,
                original_skew=original_skew,
                transformed_skew=original_skew,
                inverse_fn=lambda x: x,
                improvement=0.0,
            )

    transformed_skew = float(sp_stats.skew(transformed))
    improvement = _skew_improvement(original_skew, transformed_skew)

    def _inverse(y: np.ndarray) -> np.ndarray:
        return _yeojohnson_inverse(y, lam)

    return TransformResult(
        method=TransformMethod.YEO_JOHNSON,
        transformed=transformed,
        lambda_param=lam,
        original_skew=original_skew,
        transformed_skew=transformed_skew,
        inverse_fn=_inverse,
        improvement=improvement,
    )


def log1p_transform(v: np.ndarray) -> TransformResult | None:
    """Apply log(1 + x) transform.  Requires all values >= 0.

    Simple, fast, interpretable.  Good for right-skewed positive data
    (durations, prices, counts).

    Args:
        v: 1D array of non-negative values.

    Returns:
        TransformResult or None if data contains negative values.
    """
    if np.any(v < 0):
        return None

    original_skew = float(sp_stats.skew(v))
    transformed = np.log1p(v)
    transformed_skew = float(sp_stats.skew(transformed))
    improvement = _skew_improvement(original_skew, transformed_skew)

    return TransformResult(
        method=TransformMethod.LOG1P,
        transformed=transformed,
        lambda_param=None,
        original_skew=original_skew,
        transformed_skew=transformed_skew,
        inverse_fn=np.expm1,
        improvement=improvement,
    )


def sqrt_transform(v: np.ndarray) -> TransformResult | None:
    """Apply square root transform.  Requires all values >= 0.

    Mild variance stabilizer, good for count data (Poisson-like).

    Args:
        v: 1D array of non-negative values.

    Returns:
        TransformResult or None if data contains negative values.
    """
    if np.any(v < 0):
        return None

    original_skew = float(sp_stats.skew(v))
    transformed = np.sqrt(v)
    transformed_skew = float(sp_stats.skew(transformed))
    improvement = _skew_improvement(original_skew, transformed_skew)

    return TransformResult(
        method=TransformMethod.SQRT,
        transformed=transformed,
        lambda_param=None,
        original_skew=original_skew,
        transformed_skew=transformed_skew,
        inverse_fn=np.square,
        improvement=improvement,
    )


# ────────────────────── Auto-Selection ──────────────────────


def auto_transform(
    v: np.ndarray,
    skew_threshold: float = 0.5,
    min_improvement: float = 0.30,
) -> TransformResult:
    """Automatically select the best power transform.

    Tries all applicable transforms and picks the one that achieves the
    lowest |skewness|.  If no transform improves skewness by at least
    ``min_improvement``, returns the identity (no transform).

    Args:
        v: 1D array of numeric values (NaN/Inf should be pre-removed).
        skew_threshold: If |original_skew| < this, skip transforms entirely.
        min_improvement: Minimum improvement ratio to justify transforming.

    Returns:
        TransformResult with the best transform (or NONE if unnecessary).
    """
    v = np.asarray(v, dtype=float)
    original_skew = float(sp_stats.skew(v))

    # Already symmetric enough — no transform needed
    if abs(original_skew) < skew_threshold:
        return TransformResult(
            method=TransformMethod.NONE,
            transformed=v.copy(),
            lambda_param=None,
            original_skew=original_skew,
            transformed_skew=original_skew,
            inverse_fn=lambda x: x,
            improvement=0.0,
        )

    # Try all applicable transforms
    candidates: list[TransformResult] = []

    # Box-Cox (strictly positive only)
    bc = box_cox(v)
    if bc is not None:
        candidates.append(bc)

    # Yeo-Johnson (always applicable)
    candidates.append(yeo_johnson(v))

    # Log1p (non-negative only)
    l1p = log1p_transform(v)
    if l1p is not None:
        candidates.append(l1p)

    # Sqrt (non-negative only)
    sq = sqrt_transform(v)
    if sq is not None:
        candidates.append(sq)

    if not candidates:
        return TransformResult(
            method=TransformMethod.NONE,
            transformed=v.copy(),
            lambda_param=None,
            original_skew=original_skew,
            transformed_skew=original_skew,
            inverse_fn=lambda x: x,
            improvement=0.0,
        )

    # Pick the one with lowest |transformed_skew|
    best = min(candidates, key=lambda r: abs(r.transformed_skew))

    # Only recommend if improvement is substantial
    if best.improvement < min_improvement:
        return TransformResult(
            method=TransformMethod.NONE,
            transformed=v.copy(),
            lambda_param=None,
            original_skew=original_skew,
            transformed_skew=original_skew,
            inverse_fn=lambda x: x,
            improvement=0.0,
        )

    return best


# ────────────────────── Inverse Helpers ──────────────────────


def _boxcox_inverse(y: np.ndarray, lam: float) -> np.ndarray:
    """Inverse Box-Cox transform."""
    if abs(lam) < 1e-10:
        return np.exp(y)
    inner = y * lam + 1.0
    # Clamp to avoid negative values under fractional powers
    inner = np.maximum(inner, 1e-10)
    return np.power(inner, 1.0 / lam)


def _yeojohnson_inverse(y: np.ndarray, lam: float) -> np.ndarray:
    """Inverse Yeo-Johnson transform."""
    result = np.zeros_like(y)
    pos = y >= 0
    neg = ~pos

    if abs(lam) < 1e-10:
        result[pos] = np.exp(y[pos]) - 1.0
    else:
        inner = y[pos] * lam + 1.0
        inner = np.maximum(inner, 1e-10)
        result[pos] = np.power(inner, 1.0 / lam) - 1.0

    if abs(lam - 2.0) < 1e-10:
        result[neg] = 1.0 - np.exp(-y[neg])
    else:
        inner = -y[neg] * (2.0 - lam) + 1.0
        inner = np.maximum(inner, 1e-10)
        result[neg] = 1.0 - np.power(inner, 1.0 / (2.0 - lam))

    return result


def _skew_improvement(original: float, transformed: float) -> float:
    """Compute improvement ratio: 0 = no change, 1 = perfect symmetry."""
    orig_abs = abs(original)
    if orig_abs < 1e-10:
        return 0.0
    return max(0.0, 1.0 - abs(transformed) / orig_abs)
