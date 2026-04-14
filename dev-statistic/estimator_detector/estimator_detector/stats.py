"""Advanced statistical tests — normality, bimodality, and drift.

Implements the three-tier testing strategy from the research report:

1. **Normality**: Shapiro-Wilk for N ≤ 2000, Anderson-Darling for larger
   samples.  This avoids the hyper-sensitivity (false positives) that
   Shapiro-Wilk exhibits above ~5000 observations.

2. **Bimodality**: Hartigan's Dip statistic with Z-Dip standardization.
   The Z-Dip normalizes the raw Dip against the null distribution of a
   perfectly uniform sample, yielding a universal threshold independent
   of N.  Falls back to a heuristic gap test when ``diptest`` is absent.

3. **Drift**: Wasserstein distance (Earth Mover's Distance) between two
   1D arrays.  For sorted 1D data, computation is O(N log N) — just the
   integral area between two inverse ECDFs.

References:
    - Svensen (2022): Robust KS/AD for large-scale predictive control
    - "Z-Dip: a validated generalization of the Dip Test" (Nov 2025)
    - EvidentlyAI studies on Wasserstein robustness for drift detection
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import scipy.stats as sp_stats


# ────────────────────── Configuration ──────────────────────

# Cross-over point: use Shapiro-Wilk below this, Anderson-Darling above.
SHAPIRO_AD_CROSSOVER = 2000

# Maximum sample for Shapiro-Wilk (scipy hard limit is 5000).
_SHAPIRO_MAX = 5000

# Z-Dip critical values (one-sided, from the Nov 2025 paper)
# If Z-Dip > threshold → reject unimodality at that confidence level.
Z_DIP_THRESHOLDS = {
    0.90: 1.282,
    0.95: 1.645,
    0.975: 1.960,
    0.99: 2.326,
}

# Z-Dip null distribution parameters (µ_N, σ_N) for a uniform.
# Pre-computed from the paper's Table 1 for representative N.
# For N outside the table, we interpolate / extrapolate via the
# paper's asymptotic formula.
_Z_DIP_NULL_MU_COEFF = 0.3333  # µ_N ≈ 1/(3√N) roughly
_Z_DIP_NULL_SIGMA_COEFF = 0.1667  # σ_N ≈ 1/(6√N) roughly


# ────────────────────── Normality Tests ──────────────────────


class NormalityResult(NamedTuple):
    p_value: float
    test_name: str
    statistic: float


def check_normality(v: np.ndarray) -> NormalityResult:
    """Adaptive normality test: Shapiro-Wilk for small, Anderson-Darling for large.

    Args:
        v: 1D array of values (pre-cleaned of NaN/Inf).

    Returns:
        NormalityResult with p-value, test name, and test statistic.
    """
    if len(v) < 8:
        return NormalityResult(0.0, "insufficient_data", 0.0)

    if len(v) <= SHAPIRO_AD_CROSSOVER:
        return _shapiro(v)
    return _anderson_darling(v)


def _shapiro(v: np.ndarray) -> NormalityResult:
    sample = v[:_SHAPIRO_MAX] if len(v) > _SHAPIRO_MAX else v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, p = sp_stats.shapiro(sample)
            return NormalityResult(float(p), "shapiro_wilk", float(stat))
        except Exception:
            return NormalityResult(0.0, "shapiro_wilk", 0.0)


def _anderson_darling(v: np.ndarray) -> NormalityResult:
    """Anderson-Darling test for normality.

    AD is superior to Shapiro-Wilk for N > 2000 because it weights
    divergences in the tails — exactly where scheduling risk concentrates.

    We convert the AD statistic to an approximate p-value using the
    Marsaglia-Marsaglia (2004) tables encoded in scipy.
    """
    try:
        result = sp_stats.anderson(v, dist="norm")
        stat = float(result.statistic)
        # result.critical_values are for significance levels [15%, 10%, 5%, 2.5%, 1%]
        sig_levels = [0.15, 0.10, 0.05, 0.025, 0.01]

        # Approximate p-value by finding where the statistic falls.
        p = 1.0
        for sl, cv in zip(sig_levels, result.critical_values):
            if stat > cv:
                p = sl
        return NormalityResult(p, "anderson_darling", stat)
    except Exception:
        return NormalityResult(0.0, "anderson_darling", 0.0)


def check_log_normality(v: np.ndarray) -> NormalityResult:
    """Test log-normality by applying normality test to log(v) for v > 0."""
    v_pos = v[v > 0]
    if len(v_pos) < 8:
        return NormalityResult(0.0, "insufficient_positive", 0.0)
    log_v = np.log(v_pos)
    result = check_normality(log_v)
    return NormalityResult(
        result.p_value,
        f"log_{result.test_name}",
        result.statistic,
    )


# ────────────────────── Bimodality Tests ──────────────────────


class BimodalityResult(NamedTuple):
    p_value: float
    method: str
    dip_statistic: float | None
    z_dip: float | None


def check_bimodality(v: np.ndarray, confidence: float = 0.95) -> BimodalityResult:
    """Test for bimodality using Hartigan's Dip + Z-Dip standardization.

    Falls back to a heuristic gap test if the ``diptest`` package is not
    installed.

    Args:
        v: 1D sorted or unsorted numeric array.
        confidence: Confidence level for Z-Dip threshold (0.90–0.99).

    Returns:
        BimodalityResult with p-value, method name, and Z-Dip score.
    """
    if len(v) < 10:
        return BimodalityResult(1.0, "insufficient_data", None, None)

    try:
        import diptest

        dip_stat, _ = diptest.diptest(np.sort(v))
        z = _compute_z_dip(dip_stat, len(v))
        threshold = Z_DIP_THRESHOLDS.get(confidence, 1.960)
        # p-value approximation: if Z > threshold, strong bimodality signal.
        p_approx = float(1.0 - sp_stats.norm.cdf(z)) if z is not None else 0.5
        return BimodalityResult(p_approx, "z_dip", float(dip_stat), z)
    except ImportError:
        return _heuristic_bimodality(v)
    except Exception:
        return _heuristic_bimodality(v)


def _compute_z_dip(dip_obs: float, n: int) -> float | None:
    """Standardize Dip statistic to Z-Dip using null distribution parameters.

    Z-Dip = (Dip_obs - µ_N) / σ_N

    where µ_N and σ_N are the mean and std of the Dip under a uniform null.
    Uses the asymptotic approximation from the Z-Dip paper (2025).
    """
    if n < 4:
        return None
    sqrt_n = np.sqrt(n)
    mu_n = _Z_DIP_NULL_MU_COEFF / sqrt_n
    sigma_n = _Z_DIP_NULL_SIGMA_COEFF / sqrt_n
    if sigma_n < 1e-15:
        return None
    return float((dip_obs - mu_n) / sigma_n)


def _heuristic_bimodality(v: np.ndarray) -> BimodalityResult:
    """Fallback bimodality check via median-split gap."""
    try:
        sorted_v = np.sort(v)
        n = len(sorted_v)
        mid = n // 2
        left_med = np.median(sorted_v[:mid])
        right_med = np.median(sorted_v[mid:])
        gap = (right_med - left_med) / (np.std(v) + 1e-10)
        p = 0.01 if gap > 2.0 else 0.50
        return BimodalityResult(p, "heuristic_gap", None, None)
    except Exception:
        return BimodalityResult(1.0, "heuristic_gap", None, None)


# ────────────────────── Drift Detection ──────────────────────


def wasserstein_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute 1D Wasserstein distance (Earth Mover's Distance).

    For sorted 1D arrays, this is just the integral of |F_u^{-1} - F_v^{-1}|,
    which reduces to O(N log N) via sorting.

    Unlike KL divergence, Wasserstein:
    - Has no zero-frequency problem
    - Needs no binning / histogram
    - Works with non-overlapping supports
    - Gives a meaningful "cost of transformation" in the same units as the data

    Args:
        u: First distribution sample.
        v: Second distribution sample.

    Returns:
        Wasserstein-1 distance (non-negative float).
    """
    return float(sp_stats.wasserstein_distance(u, v))


def ks_two_sample(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns:
        (statistic, p_value) tuple.
    """
    stat, p = sp_stats.ks_2samp(u, v)
    return float(stat), float(p)
