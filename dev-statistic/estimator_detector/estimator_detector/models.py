"""estimator_detector v2 — Data models, enums, and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────── Enums ───────────────────────────


class EstimatorType(Enum):
    """Supported estimator types."""

    MEAN = "mean"
    MEDIAN = "median"
    QUANTILE_50 = "quantile_50"
    QUANTILE_75 = "quantile_75"
    QUANTILE_90 = "quantile_90"
    QUANTILE_95 = "quantile_95"
    EXPECTILE = "expectile"
    LOG_NORMAL = "log_normal"
    ENSEMBLE_QRA = "ensemble_qra"


class DistributionShape(Enum):
    """Distribution shape classification."""

    SYMMETRIC = "symmetric"
    LOG_NORMAL = "log_normal"
    SKEWED = "skewed"
    HEAVY_TAIL = "heavy_tail"
    SPARSE = "sparse"
    BIMODAL = "bimodal"
    UNKNOWN = "unknown"


class LossFamily(Enum):
    """Loss function families for gradient boosting."""

    MAE = "mae"
    MSE = "mse"
    QUANTILE = "quantile"
    EXPECTILE = "expectile"
    LINEX = "linex"
    HUBER = "huber"


class NormalityTest(Enum):
    """Which normality test was used."""

    SHAPIRO_WILK = "shapiro_wilk"
    ANDERSON_DARLING = "anderson_darling"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"


class BimodalityMethod(Enum):
    """Which bimodality test was used."""

    HEURISTIC_GAP = "heuristic_gap"
    HARTIGAN_DIP = "hartigan_dip"
    Z_DIP = "z_dip"


# ─────────────────────────── Stats ───────────────────────────


@dataclass(frozen=True)
class DistributionStats:
    """Descriptive statistics computed on the input data."""

    n: int
    mean: float
    median: float
    std: float
    cov: float
    skew: float
    kurtosis: float
    fano: float
    p25: float
    p75: float
    p90: float
    p95: float
    iqr_ratio: float
    zero_fraction: float
    median_mean_ratio: float
    # Normality testing
    normality_p: float = 0.0
    normality_test: str = "shapiro_wilk"
    log_normality_p: float = 0.0
    log_normality_test: str = "shapiro_wilk"
    # Bimodality
    bimodal_dip_p: float = 1.0
    bimodal_method: str = "heuristic_gap"
    bimodal_z_dip: float | None = None
    # Robustness
    mad: float = 0.0
    winsorized: bool = False

    # Legacy aliases
    @property
    def shapiro_p(self) -> float:
        return self.normality_p

    @property
    def log_shapiro_p(self) -> float:
        return self.log_normality_p


# ─────────────────────────── Loss Spec ───────────────────────────


@dataclass(frozen=True)
class LossSpec:
    """Fully specified loss function recommendation."""

    family: LossFamily
    catboost_loss: str
    lightgbm_objective: str
    params: dict[str, float] = field(default_factory=dict)
    has_custom_hessian: bool = False
    numba_available: bool = False
    conformal_note: str = ""

    def to_dict(self) -> dict:
        return {
            "family": self.family.value,
            "catboost_loss": self.catboost_loss,
            "lightgbm_objective": self.lightgbm_objective,
            "params": self.params,
            "has_custom_hessian": self.has_custom_hessian,
            "conformal_note": self.conformal_note,
        }


# ─────────────────────────── Recommendation ───────────────────────────


@dataclass(frozen=True)
class EstimatorRecommendation:
    """Full analysis result: which estimator, which loss, and why."""

    estimator: EstimatorType
    shape: DistributionShape
    quantile: float | None
    loss: LossSpec
    reason: str
    confidence: str
    stats: DistributionStats
    thresholds_used: dict = field(default_factory=dict)
    ensemble_weights: dict[str, float] | None = None
    drift_info: dict[str, Any] | None = None
    transform_info: dict[str, Any] | None = None

    # Convenience properties
    @property
    def catboost_loss(self) -> str:
        return self.loss.catboost_loss

    @property
    def lightgbm_objective(self) -> str:
        return self.loss.lightgbm_objective

    def to_dict(self) -> dict:
        d = {
            "estimator": self.estimator.value,
            "shape": self.shape.value,
            "quantile": self.quantile,
            "loss": self.loss.to_dict(),
            "catboost_loss": self.catboost_loss,
            "lightgbm_objective": self.lightgbm_objective,
            "reason": self.reason,
            "confidence": self.confidence,
            "stats": {
                "n": self.stats.n,
                "mean": round(self.stats.mean, 4),
                "median": round(self.stats.median, 4),
                "std": round(self.stats.std, 4),
                "cov": round(self.stats.cov, 4),
                "skew": round(self.stats.skew, 4),
                "kurtosis": round(self.stats.kurtosis, 4),
                "fano": round(self.stats.fano, 4),
                "p25": round(self.stats.p25, 4),
                "p75": round(self.stats.p75, 4),
                "p90": round(self.stats.p90, 4),
                "p95": round(self.stats.p95, 4),
                "zero_fraction": round(self.stats.zero_fraction, 4),
                "mad": round(self.stats.mad, 4),
                "normality_p": round(self.stats.normality_p, 6),
                "normality_test": self.stats.normality_test,
                "bimodal_method": self.stats.bimodal_method,
                "winsorized": self.stats.winsorized,
            },
        }
        if self.ensemble_weights:
            d["ensemble_weights"] = self.ensemble_weights
        if self.drift_info:
            d["drift_info"] = self.drift_info
        if self.transform_info:
            d["transform"] = self.transform_info
        return d

    def summary(self) -> str:
        q = f" (p{int(self.quantile * 100)})" if self.quantile else ""
        loss_tag = f" | {self.loss.family.value}"
        return f"{self.estimator.value}{q} [{self.confidence}]{loss_tag} — {self.reason}"


# ─────────────────────────── Drift Result ───────────────────────────


@dataclass(frozen=True)
class DriftResult:
    """Result from concept drift detection between two distributions."""

    drifted: bool
    distance: float
    metric: str
    threshold: float
    p_value: float | None = None
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "distance": round(self.distance, 6),
            "metric": self.metric,
            "threshold": self.threshold,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "details": self.details,
        }
