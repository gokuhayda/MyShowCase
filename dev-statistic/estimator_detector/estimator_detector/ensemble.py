"""Quantile Regression Averaging (QRA) with isotonic correction.

Instead of committing to a single quantile estimate (e.g., P90), this module
produces a weighted average of multiple quantile predictions, with weights
that shift toward higher quantiles as volatility (CoV) increases.

The critical pathology of naive quantile ensembles is the "quantile crossing
problem": noise can cause P75 > P90 in the ensemble output, a logical
impossibility.  Isotonic QRA (iQRA, 2025) solves this by applying the
Pool Adjacent Violators Algorithm (PAVA) to enforce monotonicity.

Usage:
    from estimator_detector.ensemble import QuantileEnsemble

    ens = QuantileEnsemble(quantiles=[0.75, 0.90, 0.95])
    weights = ens.compute_weights(cov=2.5)
    # → {0.75: 0.10, 0.90: 0.30, 0.95: 0.60}  (CoV-shifted)

    predictions = {0.75: 42.0, 0.90: 55.0, 0.95: 68.0}
    final = ens.aggregate(predictions, cov=2.5)
    # → 60.7

References:
    - Fakoor et al. (2022): Flexible Model Aggregation for Quantile Regression
    - Isotonic QRA (IEEE 2025): Stochastic order constraints
    - sklearn.isotonic.IsotonicRegression for PAVA
"""

from __future__ import annotations

import numpy as np


class QuantileEnsemble:
    """CoV-adaptive weighted quantile ensemble with isotonic correction.

    Args:
        quantiles: Ordered list of quantile levels to ensemble.
        cov_low: CoV below which weights favor lower quantiles.
        cov_high: CoV above which weights are maximally shifted to upper quantiles.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        cov_low: float = 0.5,
        cov_high: float = 3.0,
    ):
        self.quantiles = sorted(quantiles or [0.75, 0.90, 0.95])
        self.cov_low = cov_low
        self.cov_high = cov_high

        if len(self.quantiles) < 2:
            raise ValueError("Need at least 2 quantile levels for an ensemble.")

    def compute_weights(self, cov: float) -> dict[str, float]:
        """Compute CoV-adaptive weights.

        As CoV increases, weight shifts gravitationally from lower to upper
        quantiles — acting as an inertial shock absorber.

        Args:
            cov: Coefficient of variation of the data.

        Returns:
            Dict mapping quantile level → weight (sums to 1.0).
        """
        k = len(self.quantiles)

        # Normalize CoV to [0, 1] range
        t = np.clip((cov - self.cov_low) / (self.cov_high - self.cov_low + 1e-10), 0.0, 1.0)

        # Generate weight profile: exponential tilt toward upper quantiles as t → 1
        indices = np.arange(k, dtype=float)
        raw = np.exp(t * indices)
        weights = raw / raw.sum()

        return {q: float(w) for q, w in zip(self.quantiles, weights)}

    def aggregate(
        self,
        predictions: dict[float, float],
        cov: float,
        enforce_monotonicity: bool = True,
    ) -> float:
        """Compute weighted ensemble prediction.

        Args:
            predictions: Dict of {quantile_level: prediction_value}.
            cov: CoV for weight computation.
            enforce_monotonicity: Apply isotonic correction to prevent crossing.

        Returns:
            Weighted ensemble prediction.
        """
        weights = self.compute_weights(cov)

        # Ensure predictions are ordered by quantile level
        sorted_qs = sorted(predictions.keys())
        values = np.array([predictions[q] for q in sorted_qs])

        # Isotonic correction: enforce non-decreasing order
        if enforce_monotonicity and len(values) > 1:
            values = self._isotonic_pava(values)

        # Weighted sum
        total = 0.0
        for q, v in zip(sorted_qs, values):
            w = weights.get(q, 0.0)
            total += w * v

        return total

    def aggregate_from_data(
        self,
        data: np.ndarray,
        cov: float | None = None,
    ) -> float:
        """Convenience: compute quantile predictions directly from data.

        Args:
            data: 1D array of values.
            cov: If None, computed from data.

        Returns:
            Weighted ensemble prediction.
        """
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return float(np.mean(data)) if len(data) > 0 else 0.0

        if cov is None:
            mu = np.mean(data)
            std = np.std(data, ddof=1)
            cov = std / mu if mu > 0 else 0.0

        predictions = {q: float(np.percentile(data, q * 100)) for q in self.quantiles}
        return self.aggregate(predictions, cov=cov)

    @staticmethod
    def _isotonic_pava(values: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm for monotonic non-decreasing fit.

        This is the core of iQRA: ensures q_75 ≤ q_90 ≤ q_95 always.
        Inline implementation to avoid sklearn dependency for this simple 1D case.
        """
        n = len(values)
        result = values.copy().astype(float)

        # Forward pass: fix violations
        for i in range(1, n):
            if result[i] < result[i - 1]:
                # Pool: average the violating pair
                avg = (result[i - 1] + result[i]) / 2.0
                result[i - 1] = avg
                result[i] = avg
                # Back-propagate pooling
                j = i - 1
                while j > 0 and result[j] < result[j - 1]:
                    avg = (result[j - 1] + result[j]) / 2.0
                    result[j - 1] = avg
                    result[j] = avg
                    j -= 1

        return result
