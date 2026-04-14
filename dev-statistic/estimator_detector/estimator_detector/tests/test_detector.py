"""Tests for estimator_detector v2 — covers all modules and distribution shapes."""

import numpy as np
import pytest

from estimator_detector import (
    UniversalEstimatorDetector,
    EstimatorAdapter,
    DetectorConfig,
    DriftDetector,
    QuantileEnsemble,
    CatBoostExpectile,
    CatBoostLINEX,
    expectile_grad_hess,
    expectile_loss,
    linex_grad_hess,
    linex_loss,
    auto_transform,
    box_cox,
    yeo_johnson,
    log1p_transform,
    sqrt_transform,
    TransformMethod,
)
from estimator_detector.models import (
    DistributionShape,
    EstimatorType,
    LossFamily,
)
from estimator_detector.stats import (
    check_normality,
    check_log_normality,
    check_bimodality,
    wasserstein_distance,
)
from estimator_detector.winsorize import mad, winsorize_mad


# ━━━━━━━━━━━━━━━━━━━━━ Fixtures ━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def detector():
    return UniversalEstimatorDetector(asymmetry_ratio=3.0)


@pytest.fixture
def critical_detector():
    return UniversalEstimatorDetector(asymmetry_ratio=10.0)


@pytest.fixture
def symmetric_detector():
    return UniversalEstimatorDetector(asymmetry_ratio=1.0)


# ━━━━━━━━━━━━━━━ Distribution Classification ━━━━━━━━━━━━━━━


class TestSymmetric:
    def test_normal_data(self, detector):
        np.random.seed(42)
        data = np.random.normal(100, 5, size=200)
        rec = detector.analyze(data)
        assert rec.shape == DistributionShape.SYMMETRIC

    def test_symmetric_low_penalty_uses_mean(self, symmetric_detector):
        np.random.seed(42)
        data = np.random.normal(100, 5, size=200)
        rec = symmetric_detector.analyze(data)
        assert rec.estimator == EstimatorType.MEAN
        assert rec.catboost_loss == "MAE"

    def test_symmetric_high_penalty_uses_quantile(self, critical_detector):
        np.random.seed(42)
        data = np.random.normal(100, 5, size=200)
        rec = critical_detector.analyze(data)
        assert rec.quantile is not None
        assert rec.quantile >= 0.75


class TestLogNormal:
    def test_job_durations(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(mean=2.0, sigma=0.8, size=300)
        rec = detector.analyze(data)
        assert rec.shape == DistributionShape.LOG_NORMAL
        assert rec.quantile is not None

    def test_recommends_not_mean(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(mean=2.0, sigma=0.8, size=300)
        rec = detector.analyze(data)
        assert rec.estimator != EstimatorType.MEAN


class TestSkewed:
    def test_moderate_skew(self, detector):
        np.random.seed(42)
        data = np.random.exponential(scale=5.0, size=200)
        rec = detector.analyze(data)
        assert rec.shape in (DistributionShape.SKEWED, DistributionShape.LOG_NORMAL)
        assert rec.quantile is not None


class TestHeavyTail:
    def test_high_cov_data(self, detector):
        np.random.seed(42)
        base = np.random.exponential(scale=2.0, size=180)
        spikes = np.random.exponential(scale=50.0, size=20)
        data = np.concatenate([base, spikes])
        rec = detector.analyze(data)
        assert rec.quantile is not None
        assert rec.quantile >= 0.75

    def test_ensemble_activated(self):
        """Heavy tail + high CoV should activate QRA ensemble."""
        np.random.seed(42)
        base = np.random.exponential(scale=2.0, size=180)
        spikes = np.random.exponential(scale=100.0, size=20)
        data = np.concatenate([base, spikes])
        det = UniversalEstimatorDetector(asymmetry_ratio=5.0)
        rec = det.analyze(data)
        # Should have ensemble weights when CoV > ensemble_cov_min
        if rec.stats.cov >= 1.5:
            assert rec.ensemble_weights is not None


class TestSparse:
    def test_many_zeros(self, detector):
        np.random.seed(42)
        data = np.concatenate([np.zeros(70), np.random.exponential(10, 30)])
        rec = detector.analyze(data)
        assert rec.shape == DistributionShape.SPARSE
        assert rec.quantile == 0.90


class TestBimodal:
    def test_two_clusters(self, detector):
        np.random.seed(42)
        cluster1 = np.random.normal(10, 1, 100)
        cluster2 = np.random.normal(50, 2, 100)
        data = np.concatenate([cluster1, cluster2])
        rec = detector.analyze(data)
        assert rec.shape == DistributionShape.BIMODAL


# ━━━━━━━━━━━━━━━━━ Edge Cases ━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    def test_empty_array(self, detector):
        rec = detector.analyze(np.array([]))
        assert rec.confidence == "low"
        assert rec.stats.n == 0

    def test_single_value(self, detector):
        rec = detector.analyze(np.array([42.0]))
        assert rec.confidence == "low"

    def test_few_samples(self, detector):
        rec = detector.analyze(np.array([1, 2, 3, 4, 5]))
        assert rec.confidence == "low"
        assert rec.quantile == 0.90

    def test_all_same_value(self, detector):
        rec = detector.analyze(np.full(100, 5.0))
        assert rec.stats.std == 0.0

    def test_with_nans(self, detector):
        data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 5)
        rec = detector.analyze(data)
        assert rec.stats.n == 40

    def test_with_inf(self, detector):
        data = np.array([1, 2, np.inf, 4, 5, -np.inf, 7, 8, 9, 10] * 5)
        rec = detector.analyze(data)
        assert rec.stats.n == 40

    def test_list_input(self, detector):
        rec = detector.analyze([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)
        assert rec.stats.n == 50


# ━━━━━━━━━━━━━━━━━ Penalty / Asymmetry ━━━━━━━━━━━━━━━━━


class TestAsymmetryScaling:
    def test_higher_ratio_more_conservative(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)

        low = UniversalEstimatorDetector(asymmetry_ratio=1.0).analyze(data)
        high = UniversalEstimatorDetector(asymmetry_ratio=10.0).analyze(data)

        low_q = low.quantile or 0.0
        high_q = high.quantile or 0.0
        assert high_q >= low_q

    def test_legacy_sla_penalty_ratio(self):
        """Legacy parameter name should still work."""
        det = UniversalEstimatorDetector(sla_penalty_ratio=5.0)
        assert det.config.asymmetry_ratio == 5.0


# ━━━━━━━━━━━━━━━━━ Loss Functions ━━━━━━━━━━━━━━━━━


class TestLossFunctions:
    def test_expectile_grad_hess_shape(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.5, 2.5, 2.0, 3.0, 6.0])
        g, h = expectile_grad_hess(y, f, tau=0.8)
        assert g.shape == y.shape
        assert h.shape == y.shape
        assert np.all(h > 0)  # Hessian always positive

    def test_expectile_asymmetry(self):
        y = np.array([10.0])
        f = np.array([8.0])  # Underestimate
        g_high, _ = expectile_grad_hess(y, f, tau=0.9)
        g_low, _ = expectile_grad_hess(y, f, tau=0.1)
        # Higher tau penalizes underestimation more → larger gradient magnitude
        assert abs(g_high[0]) > abs(g_low[0])

    def test_linex_grad_hess_shape(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.5, 1.0, 4.0])
        g, h = linex_grad_hess(y, f, a=1.0)
        assert g.shape == y.shape
        assert np.all(h > 0)

    def test_linex_exponential_penalty(self):
        """Underestimation should be penalized more than overestimation."""
        y = np.array([10.0, 10.0])
        f_under = np.array([5.0, 5.0])  # Under by 5
        f_over = np.array([15.0, 15.0])  # Over by 5

        loss_under = linex_loss(y, f_under, a=1.0)
        loss_over = linex_loss(y, f_over, a=1.0)
        # With a > 0, underestimation (positive z) has exponential penalty
        assert np.mean(loss_under) > np.mean(loss_over)

    def test_expectile_loss_values(self):
        y = np.array([5.0])
        f = np.array([5.0])  # Perfect prediction
        loss = expectile_loss(y, f, tau=0.8)
        assert loss[0] == pytest.approx(0.0)

    def test_catboost_expectile_interface(self):
        obj = CatBoostExpectile(tau=0.8)
        result = obj.calc_ders_range([1.0, 2.0], [1.5, 1.0], None)
        assert len(result) == 2
        assert len(result[0]) == 2  # (grad, hess) tuple

    def test_catboost_linex_interface(self):
        obj = CatBoostLINEX(a=1.5)
        result = obj.calc_ders_range([1.0, 2.0], [1.5, 1.0], None)
        assert len(result) == 2


# ━━━━━━━━━━━━━━━━━ Statistical Tests ━━━━━━━━━━━━━━━━━


class TestNormality:
    def test_shapiro_for_small(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        result = check_normality(data)
        assert result.test_name == "shapiro_wilk"
        assert result.p_value > 0.01

    def test_anderson_for_large(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 5000)
        result = check_normality(data)
        assert result.test_name == "anderson_darling"

    def test_rejects_non_normal(self):
        np.random.seed(42)
        data = np.random.exponential(5, 500)
        result = check_normality(data)
        assert result.p_value < 0.05

    def test_log_normality(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.5, 300)
        result = check_log_normality(data)
        assert result.p_value > 0.01


class TestBimodality:
    def test_bimodal_detection(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(10, 1, 100),
            np.random.normal(50, 1, 100),
        ])
        result = check_bimodality(data)
        assert result.p_value < 0.10  # Should detect bimodality

    def test_unimodal_not_flagged(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        result = check_bimodality(data)
        assert result.p_value > 0.05


# ━━━━━━━━━━━━━━━━━ Winsorization ━━━━━━━━━━━━━━━━━


class TestWinsorize:
    def test_mad_calculation(self):
        data = np.array([1, 2, 3, 4, 5, 100])
        m = mad(data)
        assert m > 0

    def test_winsorize_small_sample(self):
        np.random.seed(42)
        data = np.concatenate([np.random.normal(10, 1, 45), [500.0]])
        processed, was_winsorized = winsorize_mad(data, threshold_n=100)
        assert was_winsorized
        assert processed.max() < 500.0

    def test_skip_large_sample(self):
        np.random.seed(42)
        data = np.random.normal(10, 1, 200)
        processed, was_winsorized = winsorize_mad(data, threshold_n=100)
        assert not was_winsorized


# ━━━━━━━━━━━━━━━━━ Drift Detection ━━━━━━━━━━━━━━━━━


class TestDrift:
    def test_no_drift_same_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(10, 2, 500)
        cur = np.random.normal(10, 2, 500)
        dd = DriftDetector(threshold=0.10)
        result = dd.compare(ref, cur)
        assert not result.drifted

    def test_detects_drift(self):
        np.random.seed(42)
        ref = np.random.normal(10, 2, 500)
        cur = np.random.normal(20, 2, 500)  # Big shift
        dd = DriftDetector(threshold=0.10)
        result = dd.compare(ref, cur)
        assert result.drifted
        assert result.distance > 0

    def test_wasserstein_distance(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([6, 7, 8, 9, 10], dtype=float)
        d = wasserstein_distance(a, b)
        assert d == pytest.approx(5.0)

    def test_drift_with_analyze(self):
        """Test drift detection through the main analyze() API."""
        np.random.seed(42)
        ref = np.random.normal(10, 2, 200)
        cur = np.random.normal(20, 2, 200)
        det = UniversalEstimatorDetector()
        rec = det.analyze(cur, reference=ref)
        assert rec.drift_info is not None
        assert rec.drift_info["drifted"]

    def test_multi_drift(self):
        np.random.seed(42)
        ref = {
            "a": np.random.normal(10, 2, 200),
            "b": np.random.normal(50, 5, 200),
        }
        cur = {
            "a": np.random.normal(10, 2, 200),  # Same
            "b": np.random.normal(80, 5, 200),  # Shifted
        }
        dd = DriftDetector(threshold=0.10)
        results = dd.compare_multi(ref, cur)
        assert not results["a"].drifted
        assert results["b"].drifted


# ━━━━━━━━━━━━━━━━━ Ensemble ━━━━━━━━━━━━━━━━━


class TestEnsemble:
    def test_weights_sum_to_one(self):
        ens = QuantileEnsemble()
        weights = ens.compute_weights(cov=2.0)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_high_cov_favors_upper(self):
        ens = QuantileEnsemble()
        low_w = ens.compute_weights(cov=0.3)
        high_w = ens.compute_weights(cov=5.0)
        # Higher CoV should put more weight on P95
        assert high_w[0.95] > low_w[0.95]

    def test_isotonic_correction(self):
        ens = QuantileEnsemble()
        # Deliberately crossed predictions: P75 > P90
        preds = {0.75: 60.0, 0.90: 50.0, 0.95: 70.0}
        result = ens.aggregate(preds, cov=2.0)
        assert isinstance(result, float)

    def test_aggregate_from_data(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 1, 500)
        ens = QuantileEnsemble()
        result = ens.aggregate_from_data(data)
        assert result > 0


# ━━━━━━━━━━━━━━━━━ Loss Recommendation ━━━━━━━━━━━━━━━━━


class TestLossRecommendation:
    def test_expectile_recommended_by_default(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        rec = detector.analyze(data)
        assert rec.loss.family == LossFamily.EXPECTILE
        assert rec.loss.has_custom_hessian

    def test_linex_for_extreme_asymmetry(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        det = UniversalEstimatorDetector(asymmetry_ratio=10.0)
        rec = det.analyze(data)
        assert rec.loss.family == LossFamily.LINEX

    def test_mae_for_symmetric_low_penalty(self, symmetric_detector):
        np.random.seed(42)
        data = np.random.normal(100, 5, 200)
        rec = symmetric_detector.analyze(data)
        assert rec.loss.family == LossFamily.MAE

    def test_conformal_note_present(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        rec = detector.analyze(data)
        assert "MapieQuantileRegressor" in rec.loss.conformal_note

    def test_lightgbm_objective(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        rec = detector.analyze(data)
        assert rec.lightgbm_objective != ""


# ━━━━━━━━━━━━━━━━━ Groups ━━━━━━━━━━━━━━━━━


class TestAnalyzeGroups:
    def test_multiple_groups(self, detector):
        np.random.seed(42)
        data = {
            "group_a": np.random.lognormal(mean=-1, sigma=0.5, size=100),
            "group_b": np.random.lognormal(mean=3, sigma=0.8, size=100),
            "group_c": np.random.lognormal(mean=4.5, sigma=1.0, size=50),
        }
        results = detector.analyze_groups(data)
        assert set(results.keys()) == {"group_a", "group_b", "group_c"}
        for rec in results.values():
            assert rec.catboost_loss is not None


# ━━━━━━━━━━━━━━━━━ Serialization ━━━━━━━━━━━━━━━━━


class TestSerialization:
    def test_to_dict(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        rec = detector.analyze(data)
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert "estimator" in d
        assert "loss" in d
        assert "stats" in d
        assert isinstance(d["stats"]["mean"], float)

    def test_summary(self, detector):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.8, 300)
        rec = detector.analyze(data)
        s = rec.summary()
        assert isinstance(s, str)
        assert len(s) > 10


# ━━━━━━━━━━━━━━━━━ Adapter ━━━━━━━━━━━━━━━━━


class TestAdapter:
    def test_local_save(self, tmp_path):
        adapter = EstimatorAdapter(output_dir=tmp_path / "configs")
        np.random.seed(42)
        result = adapter.run(
            groups={
                "demand": np.random.lognormal(-1, 0.5, 100),
                "latency": np.random.lognormal(3, 0.8, 100),
            }
        )
        assert "demand" in result
        assert "timestamp" in result
        assert (tmp_path / "configs" / "estimator_configs.json").exists()

    def test_legacy_interface(self, tmp_path):
        adapter = EstimatorAdapter(
            output_dir=tmp_path / "configs",
            rt_penalty=7.0,
            training_penalty=3.0,
        )
        np.random.seed(42)
        result = adapter.run(
            training_data={
                "group_a": np.random.lognormal(-1, 0.5, 100),
                "group_b": np.random.lognormal(3, 0.8, 100),
            }
        )
        assert "training" in result

    def test_rt_bins(self, tmp_path):
        adapter = EstimatorAdapter(output_dir=tmp_path / "configs")
        np.random.seed(42)
        rt_data = {}
        for dow in range(7):
            for hour in range(24):
                rt_data[(dow, hour)] = np.random.poisson(
                    lam=5 + 20 * (9 <= hour <= 17), size=30
                )
        result = adapter.run(rt_data=rt_data)
        assert "inference_rt" in result
        assert len(result["inference_rt"]) == 168


# ━━━━━━━━━━━━━━━━━ Performance ━━━━━━━━━━━━━━━━━


class TestPerformance:
    def test_sub_second_10k(self, detector):
        """Core requirement: 10,000 observations in < 1 second."""
        import time

        np.random.seed(42)
        data = np.random.lognormal(2, 1.5, 10_000)

        start = time.perf_counter()
        rec = detector.analyze(data)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Took {elapsed:.3f}s (limit: 1.0s)"
        assert rec.stats.n == 10_000

# ━━━━━━━━━━━━━━━━━ Transforms ━━━━━━━━━━━━━━━━━


class TestBoxCox:
    def test_positive_data(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 1.0, 500)
        result = box_cox(data)
        assert result is not None
        assert result.method == TransformMethod.BOX_COX
        assert result.lambda_param is not None
        assert abs(result.transformed_skew) < abs(result.original_skew)

    def test_rejects_negative(self):
        data = np.array([-1, 2, 3, 4, 5], dtype=float)
        result = box_cox(data)
        assert result is None

    def test_rejects_zeros(self):
        data = np.array([0, 1, 2, 3, 4], dtype=float)
        result = box_cox(data)
        assert result is None

    def test_inverse_roundtrip(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.5, 200)
        result = box_cox(data)
        assert result is not None
        recovered = result.inverse(result.transformed)
        np.testing.assert_allclose(recovered, data, rtol=1e-5)


class TestYeoJohnson:
    def test_positive_data(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 1.0, 500)
        result = yeo_johnson(data)
        assert result.method == TransformMethod.YEO_JOHNSON
        assert result.lambda_param is not None

    def test_handles_negatives(self):
        np.random.seed(42)
        data = np.random.normal(0, 5, 500)
        result = yeo_johnson(data)
        assert result.method == TransformMethod.YEO_JOHNSON

    def test_handles_zeros(self):
        np.random.seed(42)
        data = np.concatenate([np.zeros(50), np.random.exponential(5, 200)])
        result = yeo_johnson(data)
        assert result.method == TransformMethod.YEO_JOHNSON

    def test_inverse_roundtrip(self):
        np.random.seed(42)
        data = np.random.lognormal(2, 0.5, 200)
        result = yeo_johnson(data)
        recovered = result.inverse(result.transformed)
        np.testing.assert_allclose(recovered, data, rtol=1e-4)


class TestLog1p:
    def test_skewed_data(self):
        np.random.seed(42)
        data = np.random.exponential(10, 500)
        result = log1p_transform(data)
        assert result is not None
        assert result.method == TransformMethod.LOG1P
        assert abs(result.transformed_skew) < abs(result.original_skew)

    def test_rejects_negative(self):
        data = np.array([-1, 2, 3, 4, 5], dtype=float)
        result = log1p_transform(data)
        assert result is None

    def test_inverse_roundtrip(self):
        np.random.seed(42)
        data = np.random.exponential(10, 200)
        result = log1p_transform(data)
        assert result is not None
        recovered = result.inverse(result.transformed)
        np.testing.assert_allclose(recovered, data, rtol=1e-10)


class TestSqrt:
    def test_count_data(self):
        np.random.seed(42)
        data = np.random.poisson(5, 500).astype(float)
        result = sqrt_transform(data)
        assert result is not None
        assert result.method == TransformMethod.SQRT

    def test_inverse_roundtrip(self):
        np.random.seed(42)
        data = np.random.poisson(10, 200).astype(float)
        result = sqrt_transform(data)
        assert result is not None
        recovered = result.inverse(result.transformed)
        np.testing.assert_allclose(recovered, data, rtol=1e-10)


class TestAutoTransform:
    def test_selects_best(self):
        np.random.seed(42)
        data = np.random.lognormal(3, 1.5, 500)
        result = auto_transform(data)
        assert result.method != TransformMethod.NONE
        assert result.improvement > 0.3

    def test_skips_symmetric(self):
        np.random.seed(42)
        data = np.random.normal(100, 5, 500)
        result = auto_transform(data)
        assert result.method == TransformMethod.NONE

    def test_mixed_with_negatives(self):
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(-5, 2, 100),
            np.random.exponential(10, 400),
        ])
        result = auto_transform(data)
        # Should pick Yeo-Johnson (handles negatives) or NONE
        assert result.method in (TransformMethod.YEO_JOHNSON, TransformMethod.NONE)

    def test_to_dict(self):
        np.random.seed(42)
        data = np.random.lognormal(3, 1.5, 500)
        result = auto_transform(data)
        d = result.to_dict()
        assert "method" in d
        assert "improvement" in d


class TestTransformIntegration:
    def test_detector_recommends_transform(self):
        """Skewed data should include transform recommendation."""
        np.random.seed(42)
        data = np.random.lognormal(3, 1.5, 300)
        det = UniversalEstimatorDetector(asymmetry_ratio=3.0)
        rec = det.analyze(data)
        # Log-normal data should trigger a transform recommendation
        assert rec.transform_info is not None
        assert rec.transform_info["method"] != "none"

    def test_symmetric_no_transform(self):
        """Symmetric data should not recommend a transform."""
        np.random.seed(42)
        data = np.random.normal(100, 5, 300)
        det = UniversalEstimatorDetector(asymmetry_ratio=1.0)
        rec = det.analyze(data)
        assert rec.transform_info is None

    def test_transform_in_to_dict(self):
        np.random.seed(42)
        data = np.random.lognormal(3, 1.5, 300)
        det = UniversalEstimatorDetector(asymmetry_ratio=3.0)
        rec = det.analyze(data)
        d = rec.to_dict()
        if rec.transform_info:
            assert "transform" in d

    def test_transform_disabled(self):
        """Config can disable transform recommendations."""
        np.random.seed(42)
        data = np.random.lognormal(3, 1.5, 300)
        cfg = DetectorConfig(asymmetry_ratio=3.0, recommend_transform=False)
        det = UniversalEstimatorDetector(config=cfg)
        rec = det.analyze(data)
        assert rec.transform_info is None
