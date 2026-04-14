"""Universal Estimator Detector v2 — distribution analysis + loss recommendation.

Analyzes any 1D numeric array and recommends the optimal estimator and loss
function for gradient boosting (CatBoost, LightGBM, XGBoost).

Improvements over v1:
  1. Anderson-Darling for N > 2000 (no Shapiro-Wilk hyper-sensitivity)
  2. Hartigan Dip + Z-Dip for bimodality (replaces heuristic gap)
  3. Conformal prediction metadata in recommendations
  4. Expectile + LINEX losses with real Hessians (not Quantile/Pinball)
  5. Wasserstein drift detection (no KL zero-frequency problem)
  6. Isotonic QRA ensemble for volatile distributions
  7. MAD-based Winsorization for small samples (N < 100)
  8. General-purpose API — works for any numeric distribution

Example:
    from estimator_detector import UniversalEstimatorDetector

    detector = UniversalEstimatorDetector(asymmetry_ratio=3.0)
    rec = detector.analyze(data)

    print(rec.estimator)             # EstimatorType.EXPECTILE
    print(rec.catboost_loss)         # "Custom:Expectile(tau=0.80)"
    print(rec.lightgbm_objective)    # "custom:expectile(tau=0.80)"
    print(rec.loss.family)           # LossFamily.EXPECTILE
    print(rec.loss.conformal_note)   # "Wrap in MapieQuantileRegressor..."
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.stats as sp_stats

from estimator_detector.ensemble import QuantileEnsemble
from estimator_detector.models import (
    DistributionShape,
    DistributionStats,
    EstimatorRecommendation,
    EstimatorType,
    LossFamily,
    LossSpec,
)
from estimator_detector.stats import (
    check_bimodality,
    check_normality,
    check_log_normality,
)
from estimator_detector.transforms import auto_transform
from estimator_detector.winsorize import mad, winsorize_mad

logger = logging.getLogger(__name__)


# ────────────────────── Configuration ──────────────────────


@dataclass
class DetectorConfig:
    """Tunable thresholds for distribution classification.

    All defaults are empirically validated.  The only parameter most users
    need to change is ``asymmetry_ratio`` (formerly ``sla_penalty_ratio``).
    """

    # Cost asymmetry: how much worse is underestimation vs overestimation.
    # 1.0 = symmetric | 3.0 = under costs 3x | 10.0 = critical
    asymmetry_ratio: float = 3.0

    # Minimum samples for statistical analysis.
    min_samples: int = 15

    # Shape classification thresholds
    symmetric_skew_max: float = 0.5
    symmetric_normality_min: float = 0.05
    log_normal_skew_min: float = 0.5
    skewed_cov_max: float = 2.0
    skewed_skew_min: float = 1.0
    heavy_tail_cov_min: float = 2.0
    heavy_tail_kurtosis_min: float = 6.0
    heavy_tail_fano_min: float = 5.0
    sparse_zero_min: float = 0.30
    bimodal_p_threshold: float = 0.05

    # Loss family preferences
    prefer_expectile: bool = True        # Use Expectile over Quantile when applicable
    prefer_linex_above: float = 7.0      # Use LINEX when asymmetry_ratio ≥ this
    linex_a: float = 1.0                 # Default LINEX asymmetry parameter

    # Ensemble
    use_ensemble: bool = True            # Use QRA for volatile distributions
    ensemble_cov_min: float = 1.5        # CoV above which ensemble activates

    # Winsorization
    winsorize_threshold_n: int = 100     # Apply MAD-winsorize below this N
    winsorize_k: float = 3.0             # MAD fence multiplier

    # Conformal
    recommend_conformal: bool = True     # Include conformal prediction notes

    # Transforms
    recommend_transform: bool = True     # Recommend Box-Cox/Yeo-Johnson when helpful
    transform_skew_threshold: float = 0.5  # Min |skew| to consider transforms
    transform_min_improvement: float = 0.30  # Min improvement ratio to recommend

    # Legacy alias
    @property
    def sla_penalty_ratio(self) -> float:
        return self.asymmetry_ratio

    @sla_penalty_ratio.setter
    def sla_penalty_ratio(self, value: float) -> None:
        self.asymmetry_ratio = value


# ────────────────────── Detector ──────────────────────


class UniversalEstimatorDetector:
    """Analyzes any 1D numeric array and recommends estimator + loss function.

    This is a general-purpose tool — not tied to any specific domain.
    Works for response times, prices, sensor readings, durations, demand
    forecasting, or any variable with unknown distribution.

    Args:
        asymmetry_ratio: Cost asymmetry factor.  How many times worse is
            underestimating vs overestimating.  Default 3.0.
        sla_penalty_ratio: Legacy alias for asymmetry_ratio.
        min_samples: Minimum samples for reliable analysis.
        config: Full DetectorConfig for advanced tuning.
    """

    def __init__(
        self,
        asymmetry_ratio: float | None = None,
        sla_penalty_ratio: float | None = None,
        min_samples: int = 15,
        config: DetectorConfig | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            ratio = asymmetry_ratio or sla_penalty_ratio or 3.0
            self.config = DetectorConfig(
                asymmetry_ratio=ratio,
                min_samples=min_samples,
            )

        self._ensemble = QuantileEnsemble() if self.config.use_ensemble else None

    # ─── Public API ───

    def analyze(
        self,
        values: np.ndarray | list,
        reference: np.ndarray | None = None,
    ) -> EstimatorRecommendation:
        """Analyze a distribution and return a full recommendation.

        Args:
            values: 1D array of numeric values.
            reference: Optional previous distribution for drift detection.

        Returns:
            EstimatorRecommendation with estimator, loss, reason, and stats.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        # Insufficient data → conservative
        if len(values) < self.config.min_samples:
            stats = self._make_minimal_stats(values)
            return self._build(
                EstimatorType.QUANTILE_90,
                DistributionShape.UNKNOWN,
                0.90,
                self._make_loss(LossFamily.QUANTILE, alpha=0.90),
                f"Only {len(values)} samples (min: {self.config.min_samples}). "
                f"Conservative by default.",
                "low",
                stats,
            )

        # 1. Robust preprocessing for small samples
        processed, was_winsorized = winsorize_mad(
            values,
            k=self.config.winsorize_k,
            threshold_n=self.config.winsorize_threshold_n,
        )

        # 2. Compute statistics (on processed data for shape; raw for percentiles)
        stats = self._compute_stats(values, processed, was_winsorized)

        # 3. Classify shape
        shape = self._classify_shape(stats)

        # 4. Recommend estimator + loss
        rec = self._recommend(stats, shape)

        # 5. Optional transform recommendation
        transform_info = None
        if self.config.recommend_transform and shape not in (
            DistributionShape.SYMMETRIC,
            DistributionShape.SPARSE,
        ):
            tr = auto_transform(
                values,
                skew_threshold=self.config.transform_skew_threshold,
                min_improvement=self.config.transform_min_improvement,
            )
            if tr.method.value != "none":
                transform_info = tr.to_dict()

        # 6. Optional drift info
        drift_info = None
        if reference is not None:
            from estimator_detector.drift import DriftDetector

            drift_info = DriftDetector().compare(reference, values).to_dict()

        # 7. Assemble final recommendation with all optional layers
        if transform_info or drift_info:
            rec = EstimatorRecommendation(
                estimator=rec.estimator,
                shape=rec.shape,
                quantile=rec.quantile,
                loss=rec.loss,
                reason=rec.reason,
                confidence=rec.confidence,
                stats=rec.stats,
                thresholds_used=rec.thresholds_used,
                ensemble_weights=rec.ensemble_weights,
                drift_info=drift_info,
                transform_info=transform_info,
            )

        return rec

    def analyze_groups(
        self,
        data: dict[str, np.ndarray | list],
        reference: dict[str, np.ndarray] | None = None,
    ) -> dict[str, EstimatorRecommendation]:
        """Analyze multiple named groups at once.

        Args:
            data: Dict of {group_name: array_of_values}.
            reference: Optional dict of {group_name: baseline_array} for drift.

        Returns:
            Dict of {group_name: EstimatorRecommendation}.
        """
        results = {}
        for name, values in data.items():
            ref = reference.get(name) if reference else None
            results[name] = self.analyze(values, reference=ref)
        return results

    # ─── Statistics ───

    def _compute_stats(
        self,
        raw: np.ndarray,
        processed: np.ndarray,
        winsorized: bool,
    ) -> DistributionStats:
        """Compute comprehensive statistics.

        Shape metrics (skew, kurtosis) are computed on the Winsorized data
        for robustness.  Percentiles come from raw data for accuracy.
        """
        v = processed  # For shape metrics
        r = raw  # For percentiles

        mu = float(np.mean(v))
        med = float(np.median(v))
        std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

        # Normality test (adaptive: Shapiro ≤ 2000, AD > 2000)
        norm_result = check_normality(v)

        # Log-normality test
        log_norm_result = check_log_normality(v)

        # Bimodality test (Z-Dip when available, heuristic fallback)
        bimodal_result = check_bimodality(r)

        # MAD
        mad_val = mad(r)

        return DistributionStats(
            n=len(r),
            mean=mu,
            median=med,
            std=std,
            cov=std / mu if mu > 0 else 0.0,
            skew=float(sp_stats.skew(v)),
            kurtosis=float(sp_stats.kurtosis(v)),
            fano=float(np.var(v) / mu) if mu > 0 else 0.0,
            p25=float(np.percentile(r, 25)),
            p75=float(np.percentile(r, 75)),
            p90=float(np.percentile(r, 90)),
            p95=float(np.percentile(r, 95)),
            iqr_ratio=float(
                (np.percentile(r, 75) - np.percentile(r, 25)) / med
                if med > 0
                else 0.0
            ),
            zero_fraction=float(np.mean(r == 0)),
            median_mean_ratio=med / mu if mu > 0 else 1.0,
            normality_p=norm_result.p_value,
            normality_test=norm_result.test_name,
            log_normality_p=log_norm_result.p_value,
            log_normality_test=log_norm_result.test_name,
            bimodal_dip_p=bimodal_result.p_value,
            bimodal_method=bimodal_result.method,
            bimodal_z_dip=bimodal_result.z_dip,
            mad=mad_val,
            winsorized=winsorized,
        )

    def _make_minimal_stats(self, v: np.ndarray) -> DistributionStats:
        if len(v) == 0:
            return DistributionStats(
                n=0, mean=0, median=0, std=0, cov=0, skew=0, kurtosis=0,
                fano=0, p25=0, p75=0, p90=0, p95=0, iqr_ratio=0,
                zero_fraction=0, median_mean_ratio=0, normality_p=0,
                log_normality_p=0, bimodal_dip_p=1, mad=0,
            )
        mu = float(np.mean(v))
        med = float(np.median(v))
        return DistributionStats(
            n=len(v), mean=mu, median=med,
            std=float(np.std(v)) if len(v) > 1 else 0,
            cov=0, skew=0, kurtosis=0, fano=0,
            p25=float(np.percentile(v, 25)) if len(v) >= 4 else mu,
            p75=float(np.percentile(v, 75)) if len(v) >= 4 else mu,
            p90=float(np.percentile(v, 90)) if len(v) >= 10 else mu,
            p95=float(np.percentile(v, 95)) if len(v) >= 20 else mu,
            iqr_ratio=0, zero_fraction=float(np.mean(v == 0)),
            median_mean_ratio=med / mu if mu > 0 else 1,
            normality_p=0, log_normality_p=0, bimodal_dip_p=1, mad=0,
        )

    # ─── Classification ───

    def _classify_shape(self, s: DistributionStats) -> DistributionShape:
        c = self.config

        # Sparse: many zeros
        if s.zero_fraction > c.sparse_zero_min:
            return DistributionShape.SPARSE

        # Bimodal
        if s.bimodal_dip_p < c.bimodal_p_threshold:
            return DistributionShape.BIMODAL

        # Symmetric / normal
        if (
            s.normality_p > c.symmetric_normality_min
            and abs(s.skew) < c.symmetric_skew_max
        ):
            return DistributionShape.SYMMETRIC

        # Log-normal
        if (
            s.log_normality_p > c.symmetric_normality_min
            and s.skew > c.log_normal_skew_min
        ):
            return DistributionShape.LOG_NORMAL

        # Heavy tail (check before skewed — high CoV dominates)
        if (
            s.cov >= c.heavy_tail_cov_min
            or s.kurtosis > c.heavy_tail_kurtosis_min
            or s.fano > c.heavy_tail_fano_min
        ):
            return DistributionShape.HEAVY_TAIL

        # Moderate skew
        if s.cov < c.skewed_cov_max and s.skew > c.skewed_skew_min:
            return DistributionShape.SKEWED

        # Default fallback
        return DistributionShape.SKEWED

    # ─── Recommendation ───

    def _recommend(
        self, s: DistributionStats, shape: DistributionShape
    ) -> EstimatorRecommendation:
        ratio = self.config.asymmetry_ratio

        if shape == DistributionShape.SYMMETRIC:
            return self._rec_symmetric(s, ratio)
        if shape == DistributionShape.LOG_NORMAL:
            return self._rec_log_normal(s, ratio)
        if shape == DistributionShape.SKEWED:
            return self._rec_skewed(s, ratio)
        if shape == DistributionShape.HEAVY_TAIL:
            return self._rec_heavy_tail(s, ratio)
        if shape == DistributionShape.SPARSE:
            return self._rec_sparse(s, ratio)
        if shape == DistributionShape.BIMODAL:
            return self._rec_bimodal(s, ratio)

        q = self._ratio_to_quantile(ratio)
        return self._build(
            self._quantile_to_type(q), shape, q,
            self._make_loss(LossFamily.QUANTILE, alpha=q),
            "Unclassified distribution. Conservative default.", "low", s,
        )

    def _rec_symmetric(self, s, ratio):
        if ratio <= 1.5:
            return self._build(
                EstimatorType.MEAN, DistributionShape.SYMMETRIC, None,
                self._make_loss(LossFamily.MAE),
                f"Symmetric (skew={s.skew:.2f}, normality p={s.normality_p:.3f}, "
                f"test={s.normality_test}). Symmetric cost → mean is optimal.",
                "high", s,
            )
        q = 0.75 if ratio < 5 else 0.90
        tau = self._quantile_to_expectile_tau(q, ratio)
        loss = self._select_loss(ratio, q, tau)
        return self._build(
            self._quantile_to_type(q), DistributionShape.SYMMETRIC, q, loss,
            f"Symmetric but underestimation costs {ratio:.1f}x more. "
            f"Safety margin at p{int(q*100)}.",
            "high", s,
        )

    def _rec_log_normal(self, s, ratio):
        q = self._ratio_to_quantile(ratio)
        tau = self._quantile_to_expectile_tau(q, ratio)
        loss = self._select_loss(ratio, q, tau)
        return self._build(
            EstimatorType.LOG_NORMAL, DistributionShape.LOG_NORMAL, q, loss,
            f"Log-normal (median={s.median:.1f}, mean={s.mean:.1f}, "
            f"test={s.log_normality_test}). "
            f"Mean over-predicts short, under-predicts long. p{int(q*100)} corrects.",
            "high", s,
        )

    def _rec_skewed(self, s, ratio):
        q = self._ratio_to_quantile(ratio)
        tau = self._quantile_to_expectile_tau(q, ratio)
        loss = self._select_loss(ratio, q, tau)
        return self._build(
            self._quantile_to_type(q), DistributionShape.SKEWED, q, loss,
            f"Skewed (skew={s.skew:.2f}, CoV={s.cov:.2f}). "
            f"Mean under-provisions. p{int(q*100)} covers the tail.",
            "medium", s,
        )

    def _rec_heavy_tail(self, s, ratio):
        q = max(0.90, self._ratio_to_quantile(ratio))
        tau = self._quantile_to_expectile_tau(q, ratio)
        loss = self._select_loss(ratio, q, tau)

        # Ensemble for very volatile data
        ew = None
        if (
            self.config.use_ensemble
            and self._ensemble
            and s.cov >= self.config.ensemble_cov_min
        ):
            ew = self._ensemble.compute_weights(s.cov)

        est_type = (
            EstimatorType.ENSEMBLE_QRA
            if ew
            else self._quantile_to_type(q)
        )

        return self._build(
            est_type, DistributionShape.HEAVY_TAIL, q, loss,
            f"Heavy-tailed (CoV={s.cov:.2f}, kurtosis={s.kurtosis:.1f}, "
            f"fano={s.fano:.1f}). Unpredictable spikes. "
            f"{'Ensemble QRA activated. ' if ew else ''}"
            f"Aggressive margin at p{int(q*100)}.",
            "medium", s, ensemble_weights=ew,
        )

    def _rec_sparse(self, s, ratio):
        loss = self._select_loss(ratio, 0.90, 0.80)
        return self._build(
            EstimatorType.QUANTILE_90, DistributionShape.SPARSE, 0.90, loss,
            f"Sparse ({s.zero_fraction*100:.0f}% zeros). "
            f"Mean pulled down. P90 of non-zero values.",
            "low", s,
        )

    def _rec_bimodal(self, s, ratio):
        q = max(0.75, self._ratio_to_quantile(ratio))
        tau = self._quantile_to_expectile_tau(q, ratio)
        loss = self._select_loss(ratio, q, tau)
        bimodal_detail = (
            f"Z-Dip={s.bimodal_z_dip:.2f}" if s.bimodal_z_dip is not None
            else f"method={s.bimodal_method}"
        )
        return self._build(
            self._quantile_to_type(q), DistributionShape.BIMODAL, q, loss,
            f"Bimodal ({bimodal_detail}). Two clusters detected. "
            f"Mean falls in the valley between them. "
            f"p{int(q*100)} covers the upper cluster.",
            "medium", s,
        )

    # ─── Loss Function Selection ───

    def _select_loss(self, ratio: float, q: float, tau: float) -> LossSpec:
        """Select the best loss family based on asymmetry ratio and config."""
        c = self.config

        # LINEX for extreme asymmetry
        if ratio >= c.prefer_linex_above:
            return self._make_loss(LossFamily.LINEX, a=c.linex_a, q=q)

        # Expectile when preferred (default: yes)
        if c.prefer_expectile:
            return self._make_loss(LossFamily.EXPECTILE, tau=tau, q=q)

        # Fallback: Quantile
        return self._make_loss(LossFamily.QUANTILE, alpha=q)

    def _make_loss(
        self,
        family: LossFamily,
        alpha: float | None = None,
        tau: float | None = None,
        a: float | None = None,
        q: float | None = None,
    ) -> LossSpec:
        """Build a fully specified LossSpec."""
        conformal = ""
        if self.config.recommend_conformal and q:
            conformal = (
                f"Wrap model in MapieQuantileRegressor for formal "
                f"coverage guarantee at {q:.0%}. "
                f"For time-series: use Adaptive Conformal Inference (ACI)."
            )

        if family == LossFamily.MAE:
            return LossSpec(
                family=LossFamily.MAE,
                catboost_loss="MAE",
                lightgbm_objective="mae",
                conformal_note=conformal,
            )
        if family == LossFamily.MSE:
            return LossSpec(
                family=LossFamily.MSE,
                catboost_loss="RMSE",
                lightgbm_objective="regression",
                conformal_note=conformal,
            )
        if family == LossFamily.QUANTILE:
            a_ = alpha or 0.50
            return LossSpec(
                family=LossFamily.QUANTILE,
                catboost_loss=f"Quantile:alpha={a_}",
                lightgbm_objective=f"quantile",
                params={"alpha": a_},
                conformal_note=conformal,
            )
        if family == LossFamily.EXPECTILE:
            t = tau or 0.50
            return LossSpec(
                family=LossFamily.EXPECTILE,
                catboost_loss=f"Custom:Expectile(tau={t:.2f})",
                lightgbm_objective=f"custom:expectile(tau={t:.2f})",
                params={"tau": t},
                has_custom_hessian=True,
                numba_available=True,
                conformal_note=conformal,
            )
        if family == LossFamily.LINEX:
            a_val = a or 1.0
            return LossSpec(
                family=LossFamily.LINEX,
                catboost_loss=f"Custom:LINEX(a={a_val:.2f})",
                lightgbm_objective=f"custom:linex(a={a_val:.2f})",
                params={"a": a_val, "quantile_equivalent": q or 0.90},
                has_custom_hessian=True,
                numba_available=True,
                conformal_note=conformal,
            )
        if family == LossFamily.HUBER:
            return LossSpec(
                family=LossFamily.HUBER,
                catboost_loss="Huber:delta=1.0",
                lightgbm_objective="huber",
                params={"delta": 1.0},
                conformal_note=conformal,
            )

        # Fallback
        return LossSpec(
            family=family,
            catboost_loss="MAE",
            lightgbm_objective="mae",
            conformal_note=conformal,
        )

    # ─── Helpers ───

    @staticmethod
    def _ratio_to_quantile(ratio: float) -> float:
        """Map asymmetry ratio to quantile level."""
        if ratio <= 1.5:
            return 0.50
        if ratio <= 3.0:
            return 0.75
        if ratio <= 7.0:
            return 0.90
        return 0.95

    @staticmethod
    def _quantile_to_expectile_tau(q: float, ratio: float) -> float:
        """Map quantile + ratio to an Expectile τ parameter.

        Expectiles and quantiles are related but not identical.  For a
        Gaussian, τ=0.5 ↔ q=0.5.  For heavier tails, the mapping shifts.
        We use a simple heuristic: tau = q adjusted by asymmetry.
        """
        # Base: tau ≈ q
        tau = q
        # Increase tau slightly for higher ratios (more conservative)
        if ratio > 5:
            tau = min(0.98, tau + 0.05)
        return tau

    @staticmethod
    def _quantile_to_type(q: float) -> EstimatorType:
        mapping = {
            0.50: EstimatorType.QUANTILE_50,
            0.75: EstimatorType.QUANTILE_75,
            0.90: EstimatorType.QUANTILE_90,
            0.95: EstimatorType.QUANTILE_95,
        }
        return mapping.get(q, EstimatorType.QUANTILE_75)

    def _build(
        self, est, shape, q, loss, reason, conf, stats,
        ensemble_weights=None,
    ):
        return EstimatorRecommendation(
            estimator=est,
            shape=shape,
            quantile=q,
            loss=loss,
            reason=reason,
            confidence=conf,
            stats=stats,
            thresholds_used={
                "asymmetry_ratio": self.config.asymmetry_ratio,
                "min_samples": self.config.min_samples,
                "normality_test": stats.normality_test,
                "bimodal_method": stats.bimodal_method,
                "winsorized": stats.winsorized,
            },
            ensemble_weights=ensemble_weights,
        )
