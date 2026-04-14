# estimator-detector v2

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-77%20passed-green.svg)]()

**Universal distribution analysis, power transforms, and loss function recommendation for gradient boosting.**

Analyzes any 1D numeric array in **sub-second latency** (<10ms for 10k samples), classifies its distributional shape, recommends the optimal estimator, loss function (with full gradient/Hessian), and power transform — ready for CatBoost, LightGBM, and XGBoost.

> **Repository:** [github.com/gokuhayda/MyShowCase](https://github.com/gokuhayda/MyShowCase)
>
> Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
> Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Table of Contents

- [What's New in v2](#whats-new-in-v2)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The `asymmetry_ratio` Parameter](#the-asymmetry_ratio-parameter)
- [Distribution Classification](#distribution-classification)
- [Loss Function Families](#loss-function-families)
  - [Expectile Loss](#expectile-loss-default-for-ratio-15-70)
  - [LINEX Loss](#linex-loss-for-ratio--70)
  - [Why Not Quantile (Pinball)?](#why-not-quantile-pinball)
- [Power Transforms](#power-transforms)
  - [Box-Cox](#box-cox)
  - [Yeo-Johnson](#yeo-johnson)
  - [Log1p and Sqrt](#log1p-and-sqrt)
  - [Auto-Selection](#auto-selection)
- [Drift Detection](#drift-detection)
- [Quantile Ensemble (iQRA)](#quantile-ensemble-iqra)
- [Conformal Prediction](#conformal-prediction)
- [Robust Preprocessing (MAD Winsorization)](#robust-preprocessing-mad-winsorization)
- [Pipeline Adapter (DVC / CI)](#pipeline-adapter-dvc--ci)
- [Multiple Groups](#multiple-groups)
- [Full Configuration Reference](#full-configuration-reference)
- [Architecture](#architecture)
- [Tests](#tests)
- [Theoretical Foundations](#theoretical-foundations)
- [License](#license)

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Normality test | Shapiro-Wilk (hyper-sensitive > 5k) | Adaptive: Shapiro ≤ 2k, Anderson-Darling > 2k |
| Bimodality | Heuristic gap (false positives) | Hartigan Dip + Z-Dip standardization (2025) |
| Loss functions | Quantile only (zero Hessian) | Expectile + LINEX (full Newton convergence) |
| **Power transforms** | None | Box-Cox, Yeo-Johnson, log1p, sqrt + auto-select |
| Drift detection | None | Wasserstein distance (no zero-frequency problem) |
| Ensemble | Single quantile | Isotonic QRA (no crossing paradox) |
| Small samples | Raw moments (0% breakdown) | MAD Winsorization (50% breakdown point) |
| Conformal | None | MapieQuantileRegressor + ACI recommendations |
| Numba JIT | None | Optional compiled gradient/Hessian |
| LightGBM | Not supported | Full objective interface |
| Output format | CatBoost-only string | Dual CatBoost + LightGBM + LossSpec dataclass |

---

## Installation

```bash
pip install -e .              # core (numpy + scipy)
pip install -e ".[dip]"       # + Hartigan Dip test (diptest)
pip install -e ".[numba]"     # + JIT-compiled loss functions
pip install -e ".[ml]"        # + catboost + lightgbm + pandas
pip install -e ".[conformal]" # + MAPIE for conformal prediction
pip install -e ".[s3]"        # + boto3 for S3 persistence
pip install -e ".[all]"       # everything
pip install -e ".[dev]"       # + pytest + diptest for development
```

**Minimum requirements:** Python 3.10+, NumPy ≥ 1.24, SciPy ≥ 1.10.

---

## Quick Start

```python
from estimator_detector import UniversalEstimatorDetector

detector = UniversalEstimatorDetector(asymmetry_ratio=3.0)
rec = detector.analyze(data)

# Core recommendation
print(rec.estimator)              # EstimatorType.EXPECTILE
print(rec.shape)                  # DistributionShape.LOG_NORMAL
print(rec.confidence)             # "high"

# Loss function (ready for training)
print(rec.catboost_loss)          # "Custom:Expectile(tau=0.80)"
print(rec.lightgbm_objective)     # "custom:expectile(tau=0.80)"
print(rec.loss.family)            # LossFamily.EXPECTILE
print(rec.loss.has_custom_hessian) # True

# Power transform recommendation
print(rec.transform_info)
# {'method': 'box_cox', 'lambda_param': 0.234, 'original_skew': 3.21,
#  'transformed_skew': 0.08, 'improvement': 0.975}

# Conformal prediction guidance
print(rec.loss.conformal_note)
# "Wrap model in MapieQuantileRegressor for formal coverage guarantee..."

# Full summary
print(rec.summary())
# "log_normal (p75) [high] | expectile — Log-normal (median=7.4, mean=16.8)..."

# Serialization
config = rec.to_dict()  # JSON-serializable dict
```

---

## The `asymmetry_ratio` Parameter

The single parameter that adapts the package to any domain — how much worse is underestimation versus overestimation:

| Value | Meaning | Example Use Case |
|---|---|---|
| **1.0** | Symmetric cost — errors in both directions are equally bad | Cost estimation, regression baselines |
| **2.0** | Under-provisioning costs 2× more | Inventory planning |
| **3.0** | Under-provisioning costs 3× more (default) | Demand forecasting, capacity planning |
| **5.0** | Under-provisioning is significantly worse | Service latency, response time SLAs |
| **7.0** | Under-provisioning is expensive | GPU scheduling, inference SLAs |
| **10.0** | Under-provisioning is catastrophic | Safety-critical systems, OOM prevention |

```python
# Symmetric problem (e.g., price prediction)
det = UniversalEstimatorDetector(asymmetry_ratio=1.0)

# Moderate protection (e.g., demand forecasting)
det = UniversalEstimatorDetector(asymmetry_ratio=3.0)

# Critical SLA (e.g., real-time inference)
det = UniversalEstimatorDetector(asymmetry_ratio=10.0)
```

The ratio drives three decisions simultaneously: **which quantile level** to target, **which loss family** to use (MAE → Expectile → LINEX as ratio increases), and **how aggressively** the ensemble shifts weights toward upper quantiles.

---

## Distribution Classification

The detector classifies any 1D array into one of six shapes, then maps each shape to the optimal estimator strategy:

| Shape | Detection Criteria | Typical Estimator | Loss Family | Transform |
|---|---|---|---|---|
| `symmetric` | Normality p > 0.05, \|skew\| < 0.5 | mean / p50 | MAE | None |
| `log_normal` | Log-normality p > 0.05, skew > 0.5 | p75 — p90 | Expectile | Box-Cox |
| `skewed` | Moderate right skew, CoV < 2.0 | p75 | Expectile | Yeo-Johnson |
| `heavy_tail` | CoV ≥ 2.0, kurtosis > 6, or fano > 5 | p90 — p95 (QRA ensemble) | LINEX | Yeo-Johnson |
| `sparse` | > 30% zeros (intermittent demand) | p90 | Expectile | None (skipped) |
| `bimodal` | Z-Dip test / Hartigan Dip p < 0.05 | p75 — p90 | Expectile | Yeo-Johnson |

**Normality testing** adapts to sample size: Shapiro-Wilk for N ≤ 2,000 (highest power for small samples), Anderson-Darling for N > 2,000 (stable, tail-sensitive, no hyper-sensitivity false positives).

**Bimodality testing** uses Hartigan's Dip statistic with Z-Dip standardization when the `diptest` package is installed, providing a universal threshold independent of sample size. Falls back to a heuristic gap test otherwise.

---

## Loss Function Families

### Expectile Loss (default for ratio 1.5–7.0)

The **quadratic asymmetric** counterpart of Quantile regression. Key advantage: a **non-zero Hessian** that enables full Newton-Raphson convergence in CatBoost/LightGBM, converging orders of magnitude faster than Quantile.

```python
from estimator_detector import CatBoostExpectile, expectile_grad_hess

# Direct vectorized computation
grad, hess = expectile_grad_hess(y_true, y_pred, tau=0.80)
# grad.shape == hess.shape == y_true.shape
# hess > 0 everywhere (full Newton, no collapse)

# CatBoost integration
from catboost import CatBoostRegressor
model = CatBoostRegressor(loss_function=CatBoostExpectile(tau=0.80))
model.fit(X_train, y_train)

# LightGBM integration
from estimator_detector.losses import lgbm_expectile_objective
import lightgbm as lgb
model = lgb.train(
    params={},
    train_set=lgb.Dataset(X_train, y_train),
    fobj=lgbm_expectile_objective(tau=0.80),
)
```

The τ parameter controls asymmetry: τ > 0.5 penalizes underestimation more heavily (analogous to targeting a quantile above the median). The detector maps `asymmetry_ratio` to an appropriate τ value automatically.

### LINEX Loss (for ratio ≥ 7.0)

**Linear-Exponential loss** — the gold standard for extreme asymmetric cost scenarios. Underestimation is penalized **exponentially**, overestimation **linearly**.

```python
from estimator_detector import CatBoostLINEX, linex_grad_hess

# Direct computation
grad, hess = linex_grad_hess(y_true, y_pred, a=1.5)
# Underestimation (y > f): exponential penalty
# Overestimation (y < f): linear penalty

# CatBoost integration
model = CatBoostRegressor(loss_function=CatBoostLINEX(a=1.5))
model.fit(X_train, y_train)
```

The `a` parameter controls exponential severity. For SLA-critical scenarios where a single underestimate causes catastrophic failure, LINEX ensures the model learns to err on the side of overestimation.

### Why Not Quantile (Pinball)?

The standard Quantile loss is **piecewise linear** — its second derivative (Hessian) is **zero everywhere**. CatBoost and LightGBM use second-order Newton-Raphson optimization, so when the Hessian is zero, the Newton step `-(g/h)` collapses to unscaled gradient descent. This causes extremely slow convergence, requiring many more boosting rounds to achieve the same accuracy.

Both Expectile and LINEX are **strictly convex and twice-differentiable**, providing real Hessian values that enable full Newton acceleration.

---

## Power Transforms

When the target variable is highly skewed, applying a power transform before training can substantially improve model convergence and residual symmetry. The detector automatically recommends the best transform for each distribution shape.

### Box-Cox

Classical power transform for **strictly positive** data. Finds λ that maximizes normality of the transformed distribution.

```python
from estimator_detector import box_cox

result = box_cox(data)  # Returns None if data contains zeros/negatives
print(result.method)           # TransformMethod.BOX_COX
print(result.lambda_param)     # e.g., 0.234
print(result.original_skew)    # e.g., 3.21
print(result.transformed_skew) # e.g., 0.08
print(result.improvement)      # e.g., 0.975 (97.5% skewness reduction)

# Back-transform predictions
predictions_original_scale = result.inverse(predictions_transformed)
```

### Yeo-Johnson

Generalization of Box-Cox that **handles zero and negative values**. Always applicable — preferred when data contains zeros (sparse/count distributions) or mixed-sign values.

```python
from estimator_detector import yeo_johnson

result = yeo_johnson(data)  # Works with any real-valued data
print(result.lambda_param)  # Yeo-Johnson λ
recovered = result.inverse(result.transformed)  # Roundtrip
```

### Log1p and Sqrt

Simpler, faster transforms for specific cases:

```python
from estimator_detector import log1p_transform, sqrt_transform

# log(1 + x) — fast, interpretable, good for right-skewed positive data
log_result = log1p_transform(data)

# √x — mild variance stabilizer for count data (Poisson-like)
sqrt_result = sqrt_transform(data)
```

### Auto-Selection

`auto_transform` tries all applicable transforms and picks the one that achieves the lowest |skewness|, with a configurable minimum improvement threshold:

```python
from estimator_detector import auto_transform

result = auto_transform(data, skew_threshold=0.5, min_improvement=0.30)

if result.method.value != "none":
    print(f"Recommended: {result.method.value} (λ={result.lambda_param})")
    print(f"Skewness: {result.original_skew:.2f} → {result.transformed_skew:.2f}")
    print(f"Improvement: {result.improvement:.1%}")
    
    # Use transformed data for training
    model.fit(X_train, result.transformed)
    
    # Back-transform predictions
    y_pred = result.inverse(model.predict(X_test))
```

**Integration with the detector:** when `recommend_transform=True` (default), the detector includes transform recommendations in `rec.transform_info` for any non-symmetric distribution:

```python
rec = detector.analyze(data)
if rec.transform_info:
    print(rec.transform_info)
    # {'method': 'yeo_johnson', 'lambda_param': 0.42, 
    #  'original_skew': 2.87, 'transformed_skew': 0.12, 'improvement': 0.96}
```

---

## Drift Detection

Wasserstein-based (Earth Mover's Distance) concept drift detection — no binning, no zero-frequency problems, O(N log N):

```python
from estimator_detector import DriftDetector

dd = DriftDetector(threshold=0.10)  # 10% of IQR
result = dd.compare(last_week_data, this_week_data)

print(result.drifted)     # True/False — ready for CI/CD pipelines
print(result.distance)    # Wasserstein distance in data units
print(result.details)     # "Wasserstein=0.42, threshold=0.38, KS p=0.001"

if result.drifted:
    trigger_retrain()
```

**Why Wasserstein over KL/PSI?**

- **KL divergence** explodes to ∞ when a new value appears that never occurred in the reference (zero-frequency problem) — common in continuous data
- **PSI** requires binning, and bin count is a hyperparameter that induces false positives
- **Wasserstein** is continuous, bin-free, and gives a meaningful "cost of transformation" in the same units as the data

**Multi-group drift:**

```python
results = dd.compare_multi(
    reference={"demand": last_week_demand, "latency": last_week_latency},
    current={"demand": this_week_demand, "latency": this_week_latency},
)
for name, result in results.items():
    print(f"{name}: {'DRIFT' if result.drifted else 'stable'} ({result.distance:.4f})")
```

**Integrated with `analyze()`:**

```python
rec = detector.analyze(current_data, reference=last_week_data)
print(rec.drift_info)  # {"drifted": True, "distance": 0.42, ...}
```

---

## Quantile Ensemble (iQRA)

For highly volatile distributions (CoV > 1.5), committing to a single quantile estimate (e.g., P90) is fragile. The **Isotonic Quantile Regression Averaging** ensemble:

1. Weights multiple quantile predictions (P75, P90, P95)
2. Shifts weights toward upper quantiles as CoV increases (shock absorber)
3. Applies **PAVA (Pool Adjacent Violators Algorithm)** to enforce monotonicity — preventing the quantile crossing paradox (P75 > P90)

```python
from estimator_detector import QuantileEnsemble

ens = QuantileEnsemble(quantiles=[0.75, 0.90, 0.95])

# CoV-adaptive weights
weights_low = ens.compute_weights(cov=0.3)   # {0.75: 0.45, 0.90: 0.33, 0.95: 0.22}
weights_high = ens.compute_weights(cov=3.0)  # {0.75: 0.08, 0.90: 0.27, 0.95: 0.65}

# Aggregate predictions
prediction = ens.aggregate(
    predictions={0.75: 42.0, 0.90: 55.0, 0.95: 68.0},
    cov=2.5,
)

# Or directly from data
prediction = ens.aggregate_from_data(data)
```

The detector activates the ensemble automatically for heavy-tailed distributions and includes the weights in `rec.ensemble_weights`.

---

## Conformal Prediction

When `recommend_conformal=True` (default), every recommendation includes guidance on wrapping the model in a **conformal prediction** framework for formal coverage guarantees:

```python
rec = detector.analyze(data)
print(rec.loss.conformal_note)
# "Wrap model in MapieQuantileRegressor for formal coverage guarantee at 75%.
#  For time-series: use Adaptive Conformal Inference (ACI)."
```

This tells you to:
1. Train with the recommended loss (Expectile/LINEX)
2. Wrap in `mapie.regression.MapieQuantileRegressor` for calibrated prediction intervals
3. For time-series data, use **Adaptive Conformal Inference (ACI)** which dynamically adjusts the significance level as coverage violations accumulate

---

## Robust Preprocessing (MAD Winsorization)

For arrays with N < 100, traditional moment statistics (skewness, kurtosis, CoV) have a **breakdown point of 0%** — a single corrupted observation can shift all higher-order moments to infinity, causing the detector to misclassify a stable signal as "heavy-tailed extreme."

The **Median Absolute Deviation (MAD)**, scaled by 1.4826 for Gaussian consistency, has a **breakdown point of 50%**. Before shape classification, small arrays are conditionally Winsorized: extreme values are clamped to [median ± 3·MAD] fences, preserving sample size while preventing outlier-driven distortion.

```python
from estimator_detector.winsorize import mad, winsorize_mad

# MAD (scaled) — robust scale estimator
scale = mad(data)  # ≈ σ for Gaussian data, robust to outliers

# Conditional Winsorization
processed, was_clipped = winsorize_mad(data, k=3.0, threshold_n=100)
# Only applied for N < 100; large samples are returned unchanged
```

This is applied automatically inside `analyze()` — no user action needed.

---

## Pipeline Adapter (DVC / CI)

The `EstimatorAdapter` wraps the detector for periodic pipeline execution with persistence and drift tracking:

```python
from estimator_detector import EstimatorAdapter

adapter = EstimatorAdapter(
    output_dir="configs/",
    s3_bucket="my-bucket",              # optional S3 persistence
    s3_prefix="estimator_configs/",
    default_ratio=3.0,
    group_ratios={                      # per-group asymmetry
        "latency": 7.0,
        "cost": 1.0,
        "demand": 3.0,
    },
    drift_threshold=0.10,               # Wasserstein IQR-relative
)

# Run analysis with drift detection
result = adapter.run(
    groups={
        "demand": current_demand_data,
        "latency": current_latency_data,
        "cost": current_cost_data,
    },
    reference={
        "demand": last_week_demand,
        "latency": last_week_latency,
        "cost": last_week_cost,
    },
)
# → Saves configs/estimator_configs.json + S3 backup with timestamp
```

**DVC integration example:**

```yaml
# dvc.yaml
stages:
  detect_estimators:
    cmd: python scripts/detect.py
    deps:
      - data/processed/metrics.parquet
    outs:
      - configs/estimator_configs.json
```

---

## Multiple Groups

Analyze multiple named distributions in a single call:

```python
results = detector.analyze_groups({
    "demand": demand_data,
    "latency": latency_data,
    "cost": cost_data,
})

for name, rec in results.items():
    print(f"{name}: {rec.catboost_loss}")
    print(f"  Shape: {rec.shape.value}")
    print(f"  Transform: {rec.transform_info}")
    print(f"  Reason: {rec.reason}")
```

With drift detection:

```python
results = detector.analyze_groups(
    data=current_data,
    reference=last_week_data,
)
for name, rec in results.items():
    drift = "DRIFT" if rec.drift_info and rec.drift_info["drifted"] else "stable"
    print(f"{name}: {rec.catboost_loss} [{drift}]")
```

---

## Full Configuration Reference

```python
from estimator_detector import DetectorConfig, UniversalEstimatorDetector

config = DetectorConfig(
    # Core
    asymmetry_ratio=3.0,          # Cost asymmetry (1.0 = symmetric, 10.0 = critical)
    min_samples=15,               # Minimum samples for reliable analysis
    
    # Shape classification thresholds
    symmetric_skew_max=0.5,       # |skew| < this → symmetric
    symmetric_normality_min=0.05, # normality p > this → not rejected
    log_normal_skew_min=0.5,      # skew > this + log-normality p > 0.05
    skewed_cov_max=2.0,           # CoV < this + skew > 1.0 → skewed
    skewed_skew_min=1.0,
    heavy_tail_cov_min=2.0,       # CoV ≥ this → heavy tail
    heavy_tail_kurtosis_min=6.0,  # kurtosis > this → heavy tail
    heavy_tail_fano_min=5.0,      # fano > this → heavy tail
    sparse_zero_min=0.30,         # > 30% zeros → sparse
    bimodal_p_threshold=0.05,     # dip p < this → bimodal
    
    # Loss preferences
    prefer_expectile=True,        # Use Expectile over Quantile
    prefer_linex_above=7.0,       # Use LINEX when ratio ≥ this
    linex_a=1.0,                  # LINEX asymmetry parameter
    
    # Ensemble
    use_ensemble=True,            # Activate QRA for volatile distributions
    ensemble_cov_min=1.5,         # CoV threshold for ensemble activation
    
    # Winsorization
    winsorize_threshold_n=100,    # Apply MAD-winsorize below this N
    winsorize_k=3.0,              # MAD fence multiplier
    
    # Transforms
    recommend_transform=True,     # Recommend Box-Cox/Yeo-Johnson
    transform_skew_threshold=0.5, # Min |skew| to consider transforms
    transform_min_improvement=0.30, # Min improvement to recommend
    
    # Conformal
    recommend_conformal=True,     # Include conformal prediction notes
)

detector = UniversalEstimatorDetector(config=config)
```

---

## Architecture

```
estimator_detector/
├── __init__.py        # Public API — all exports
├── models.py          # Enums, dataclasses: EstimatorType, LossSpec, DriftResult
├── detector.py        # UniversalEstimatorDetector — core classification engine
├── stats.py           # AD/KS normality, Z-Dip bimodality, Wasserstein
├── losses.py          # Expectile, LINEX: vectorized grad/hess + CatBoost/LightGBM
├── transforms.py      # Box-Cox, Yeo-Johnson, log1p, sqrt, auto_transform
├── drift.py           # DriftDetector — Wasserstein-based concept drift
├── ensemble.py        # QuantileEnsemble — Isotonic QRA with PAVA
├── winsorize.py       # MAD-based robust preprocessing
├── adapter.py         # EstimatorAdapter — pipeline persistence (disk + S3)
└── tests/
    └── test_detector.py  # 77 tests covering all modules
```

**Data flow:**

```
Input array
  → [Winsorize if N < 100]
  → [Compute stats: normality, bimodality, moments]
  → [Classify shape]
  → [Select estimator + quantile]
  → [Select loss family: MAE / Expectile / LINEX]
  → [Recommend transform: Box-Cox / Yeo-Johnson / None]
  → [Optional: drift detection vs reference]
  → [Optional: QRA ensemble weights]
  → EstimatorRecommendation
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

**77 tests** covering:

- Distribution classification (symmetric, log-normal, skewed, heavy-tail, sparse, bimodal)
- Edge cases (empty, single value, NaN, Inf, all-same, list input)
- Asymmetry scaling and legacy API compatibility
- Loss functions (Expectile, LINEX: gradients, Hessians, CatBoost/LightGBM interfaces)
- Normality tests (Shapiro-Wilk, Anderson-Darling, adaptive crossover)
- Bimodality detection (Z-Dip, heuristic fallback)
- MAD Winsorization (small/large samples, edge cases)
- Drift detection (no-drift, drift, multi-group, integrated)
- Quantile ensemble (weight computation, isotonic correction, data aggregation)
- Power transforms (Box-Cox, Yeo-Johnson, log1p, sqrt, auto-select, inverse roundtrips)
- Transform integration (detector recommendations, disabled config)
- Serialization (to_dict, summary)
- Pipeline adapter (local save, legacy interface, RT bins)
- Performance (10k samples in < 1 second)

---

## Theoretical Foundations

This package synthesizes research from eight domains:

1. **Normality testing at scale** — Anderson-Darling for N > 2k avoids Shapiro-Wilk's hyper-sensitivity to micro-noise in large samples (Svensen, 2022)

2. **Bimodality detection** — Z-Dip standardization of Hartigan's Dip test provides a universal threshold independent of N, eliminating bootstrap overhead ("Z-Dip: a validated generalization of the Dip Test", Nov 2025)

3. **Asymmetric loss functions** — Expectile regression (Toth 2016, Laimighofer 2022) and LINEX loss (Shrivastava et al. 2023) provide non-zero Hessians for full second-order Newton convergence in gradient boosting

4. **Power transforms** — Box-Cox (Box & Cox, 1964) and Yeo-Johnson (Yeo & Johnson, 2000) normalize skewed targets before training, improving residual symmetry and model convergence

5. **Conformal prediction** — Split Conformal / CQR provides mathematically guaranteed coverage intervals without distributional assumptions (Gibbs & Candès 2021/2022, Zaffran et al. 2022)

6. **Drift detection** — Wasserstein distance eliminates the zero-frequency problem of KL divergence and the bin-dependency of PSI (EvidentlyAI, 2023)

7. **Quantile ensemble** — Isotonic QRA with PAVA prevents the quantile crossing paradox while providing CoV-adaptive weight shifting (Fakoor et al. 2022, Isotonic QRA IEEE 2025)

8. **Robust preprocessing** — MAD-based Winsorization has a 50% breakdown point vs 0% for raw moments, critical for micro-batch analysis (Clark 1995/2017, Cheng & Young 2023)

---

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix or transform, you must distribute under the same license

Full license text: [creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
