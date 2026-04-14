"""Robust preprocessing — MAD-based winsorization for outlier resilience.

For arrays with N < 100, traditional moment statistics (skewness, kurtosis, CoV)
have a breakdown point of 0%.  A single corrupted observation can shift all
higher-order moments to infinity.

The Median Absolute Deviation (MAD), scaled by 1.4826 for Gaussian consistency,
has a breakdown point of 50% — making it ideal as a guard before shape
classification.  Rather than trimming (which discards data and destroys
confidence intervals), we Winsorize: clamp extreme values to the fences.

References:
    - Clark (1995/2017) on optimal Winsorization cutoffs
    - Cheng & Young (2023) on micro-series distortion in MLOps
"""

from __future__ import annotations

import numpy as np


# Scaling factor so MAD ≈ σ for Gaussian data
_MAD_SCALE = 1.4826


def mad(v: np.ndarray) -> float:
    """Compute the scaled Median Absolute Deviation.

    MAD = median(|X_i - median(X)|) × 1.4826

    The scale factor ensures MAD ≈ σ for normally distributed data.

    Args:
        v: 1D array of numeric values (NaN/Inf should already be removed).

    Returns:
        Scaled MAD value.  Returns 0.0 for arrays with fewer than 3 elements.
    """
    if len(v) < 3:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)) * _MAD_SCALE)


def winsorize_mad(
    v: np.ndarray,
    k: float = 3.0,
    threshold_n: int = 100,
) -> tuple[np.ndarray, bool]:
    """Conditionally Winsorize an array using MAD fences.

    If len(v) >= threshold_n, the array is returned unchanged — large samples
    are robust enough for raw moment estimation.

    For smaller arrays, values are clamped to [median - k*MAD, median + k*MAD].
    This preserves sample size (unlike trimming) while preventing outlier-driven
    distortion of skewness/kurtosis.

    Args:
        v: 1D numeric array.
        k: Number of MAD units for the fence.  Default 3.0 ≈ 3σ for Gaussians.
        threshold_n: Skip Winsorization for arrays this size or larger.

    Returns:
        (processed_array, was_winsorized) tuple.
    """
    if len(v) < 3 or len(v) >= threshold_n:
        return v, False

    m = mad(v)
    if m < 1e-12:
        # All values are (nearly) identical — nothing to clamp.
        return v, False

    med = np.median(v)
    lo = med - k * m
    hi = med + k * m

    clipped = np.clip(v, lo, hi)
    changed = not np.array_equal(v, clipped)
    return clipped, changed
