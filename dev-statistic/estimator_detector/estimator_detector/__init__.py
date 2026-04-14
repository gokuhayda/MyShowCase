"""estimator_detector v2 — Universal distribution analysis + loss recommendation.

Analyzes any 1D numeric array, classifies its shape, recommends the optimal
estimator, loss function, and power transform for gradient boosting frameworks.

Quick start:
    >>> from estimator_detector import UniversalEstimatorDetector
    >>> rec = UniversalEstimatorDetector(asymmetry_ratio=3.0).analyze(data)
    >>> print(rec.catboost_loss, rec.lightgbm_objective)
    >>> print(rec.transform_info)  # Box-Cox / Yeo-Johnson recommendation

Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
Licensed under CC BY-NC-SA 4.0.
"""

from estimator_detector.adapter import EstimatorAdapter
from estimator_detector.detector import DetectorConfig, UniversalEstimatorDetector
from estimator_detector.drift import DriftDetector
from estimator_detector.ensemble import QuantileEnsemble
from estimator_detector.losses import (
    CatBoostExpectile,
    CatBoostLINEX,
    expectile_grad_hess,
    expectile_loss,
    linex_grad_hess,
    linex_loss,
)
from estimator_detector.models import (
    BimodalityMethod,
    DistributionShape,
    DistributionStats,
    DriftResult,
    EstimatorRecommendation,
    EstimatorType,
    LossFamily,
    LossSpec,
    NormalityTest,
)
from estimator_detector.transforms import (
    TransformMethod,
    TransformResult,
    auto_transform,
    box_cox,
    yeo_johnson,
    log1p_transform,
    sqrt_transform,
)

__version__ = "2.0.0"

__all__ = [
    # Core
    "UniversalEstimatorDetector",
    "DetectorConfig",
    "EstimatorAdapter",
    # Models
    "EstimatorType",
    "DistributionShape",
    "DistributionStats",
    "EstimatorRecommendation",
    "LossFamily",
    "LossSpec",
    "DriftResult",
    "NormalityTest",
    "BimodalityMethod",
    # Drift
    "DriftDetector",
    # Ensemble
    "QuantileEnsemble",
    # Losses
    "CatBoostExpectile",
    "CatBoostLINEX",
    "expectile_grad_hess",
    "expectile_loss",
    "linex_grad_hess",
    "linex_loss",
    # Transforms
    "TransformMethod",
    "TransformResult",
    "auto_transform",
    "box_cox",
    "yeo_johnson",
    "log1p_transform",
    "sqrt_transform",
]
