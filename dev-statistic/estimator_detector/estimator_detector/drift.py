"""Concept drift detection via Wasserstein distance.

Detects whether a distribution has shifted significantly between two time
windows (e.g., last week vs. this week).

Why Wasserstein over KL/PSI:
- No zero-frequency problem (KL divergence → ∞ when P(x)=0, Q(x)>0)
- No binning required (PSI performance depends on bin count choice)
- Continuous metric in the same units as the data
- O(N log N) for 1D arrays (just sorting + area between ECDFs)

The DriftDetector class provides:
- A boolean ``drifted`` flag for CI/CD pipelines
- The raw Wasserstein distance for monitoring dashboards
- Optional KS two-sample test as a secondary signal

References:
    - EvidentlyAI (2023): Wasserstein robustness for large-scale drift detection
"""

from __future__ import annotations

import numpy as np
import scipy.stats as sp_stats

from estimator_detector.models import DriftResult


class DriftDetector:
    """Detect concept drift between two numeric distributions.

    Usage:
        dd = DriftDetector(threshold=0.10)
        result = dd.compare(last_week_data, this_week_data)
        if result.drifted:
            trigger_retrain()

    The threshold is relative: it is expressed as a fraction of the reference
    distribution's IQR.  This makes it scale-invariant — a threshold of 0.10
    means "the distributions shifted by more than 10% of the reference IQR".

    For absolute thresholds, set ``relative=False`` and provide the threshold
    in the original data units.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        relative: bool = True,
        ks_alpha: float = 0.05,
    ):
        """
        Args:
            threshold: Drift threshold.  Relative to IQR by default.
            relative: If True, threshold is IQR-relative.  If False, absolute.
            ks_alpha: Significance level for the supplementary KS test.
        """
        self.threshold = threshold
        self.relative = relative
        self.ks_alpha = ks_alpha

    def compare(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Compare two distributions and report drift.

        Args:
            reference: Baseline / historical distribution.
            current: New / current distribution.

        Returns:
            DriftResult with drifted flag, distance, and details.
        """
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 5 or len(cur) < 5:
            return DriftResult(
                drifted=False,
                distance=0.0,
                metric="wasserstein",
                threshold=self.threshold,
                details="Insufficient data for drift detection "
                f"(ref={len(ref)}, cur={len(cur)}).",
            )

        w_dist = float(sp_stats.wasserstein_distance(ref, cur))

        # Determine the effective threshold
        if self.relative:
            iqr = float(np.percentile(ref, 75) - np.percentile(ref, 25))
            if iqr < 1e-12:
                iqr = float(np.std(ref)) or 1.0
            effective_threshold = self.threshold * iqr
        else:
            effective_threshold = self.threshold

        drifted = w_dist > effective_threshold

        # Supplementary KS test
        ks_stat, ks_p = sp_stats.ks_2samp(ref, cur)

        details = (
            f"Wasserstein={w_dist:.4f}, "
            f"threshold={effective_threshold:.4f} "
            f"({'relative' if self.relative else 'absolute'}), "
            f"KS stat={ks_stat:.4f}, KS p={ks_p:.4f}"
        )

        return DriftResult(
            drifted=drifted,
            distance=w_dist,
            metric="wasserstein",
            threshold=effective_threshold,
            p_value=float(ks_p),
            details=details,
        )

    def compare_multi(
        self,
        reference: dict[str, np.ndarray],
        current: dict[str, np.ndarray],
    ) -> dict[str, DriftResult]:
        """Compare multiple named distributions at once.

        Args:
            reference: Dict of {name: baseline_array}.
            current: Dict of {name: current_array}.

        Returns:
            Dict of {name: DriftResult} for each common key.
        """
        results = {}
        common_keys = set(reference.keys()) & set(current.keys())
        for key in sorted(common_keys):
            results[key] = self.compare(reference[key], current[key])
        return results
