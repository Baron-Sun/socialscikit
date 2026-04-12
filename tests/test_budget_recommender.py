"""Tests for socialscikit.quantikit.budget_recommender."""

import pytest

from socialscikit.quantikit.budget_recommender import (
    BudgetRecommender,
    BudgetReport,
    _find_n_for_target,
    _fit_power_law,
    _marginal_curve,
    _power_law,
    _predict_f1,
)
from socialscikit.quantikit.feature_extractor import TaskFeatures

import numpy as np


@pytest.fixture()
def recommender():
    return BudgetRecommender()


# Realistic historical points (simulated learning curve)
HISTORICAL_POINTS = [
    (50, 0.55), (100, 0.63), (200, 0.70), (300, 0.74),
    (500, 0.79), (800, 0.82), (1000, 0.84),
]


# ---------------------------------------------------------------------------
# Power-law primitives
# ---------------------------------------------------------------------------


class TestPowerLaw:
    def test_power_law_monotonic(self):
        """f1 should increase with n."""
        ns = np.array([100, 500, 1000, 5000], dtype=float)
        vals = _power_law(ns, 0.1, 0.3, 0.3)
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_fit_recovers_trend(self):
        """Fitting should produce a reasonable curve on known data."""
        points = [(100, 0.60), (300, 0.72), (500, 0.78), (1000, 0.84)]
        a, b, c = _fit_power_law(points)
        # Predictions should be close to observed
        for n, f1_obs in points:
            f1_pred = _predict_f1(n, a, b, c)
            assert abs(f1_pred - f1_obs) < 0.05, f"n={n}: pred={f1_pred:.3f} vs obs={f1_obs}"

    def test_fit_with_single_point_is_unreliable(self):
        """Single point produces a fit but it's unreliable (no real constraint)."""
        # curve_fit won't raise, but the result is meaningless
        a, b, c = _fit_power_law([(100, 0.5)])
        # Should still return numbers (degenerate but valid)
        assert isinstance(a, float)

    def test_fit_fails_on_empty(self):
        """Empty input should fail."""
        with pytest.raises((ValueError, TypeError)):
            _fit_power_law([])

    def test_predict_capped_at_1(self):
        """F1 should never exceed 1.0."""
        assert _predict_f1(999999, 10.0, 0.5, 0.0) <= 1.0


class TestFindN:
    def test_finds_reasonable_n(self):
        a, b, c = _fit_power_law(HISTORICAL_POINTS)
        n = _find_n_for_target(0.80, a, b, c)
        assert n is not None
        assert 200 < n < 2000

    def test_unreachable_target_returns_none(self):
        a, b, c = _fit_power_law(HISTORICAL_POINTS)
        n = _find_n_for_target(0.999, a, b, c, max_n=5000)
        # May or may not be reachable; if not, returns None
        # The curve fitted from our points plateaus around 0.88-0.90
        # so 0.999 may be unreachable
        if n is not None:
            assert n > 1000


class TestMarginalCurve:
    def test_curve_length(self):
        a, b, c = _fit_power_law(HISTORICAL_POINTS)
        curve = _marginal_curve(a, b, c, max_n=2000, step=100)
        assert len(curve) == 20  # 2000/100

    def test_curve_monotonic(self):
        a, b, c = _fit_power_law(HISTORICAL_POINTS)
        curve = _marginal_curve(a, b, c, max_n=3000, step=100)
        f1_values = [f1 for _, f1 in curve]
        assert all(f1_values[i] <= f1_values[i + 1] for i in range(len(f1_values) - 1))

    def test_curve_values_bounded(self):
        a, b, c = _fit_power_law(HISTORICAL_POINTS)
        curve = _marginal_curve(a, b, c)
        for _, f1 in curve:
            assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Empirical mode
# ---------------------------------------------------------------------------


class TestEmpiricalMode:
    def test_basic_empirical(self, recommender):
        features = TaskFeatures(n_labeled=1000, target_f1=0.85)
        report = recommender.recommend(
            features, historical_points=HISTORICAL_POINTS, target_f1=0.85,
        )
        assert report.estimation_basis == "empirical"
        assert report.recommended_n > 0
        assert report.prior_source is None
        assert report.fitting_params is not None
        assert "a" in report.fitting_params

    def test_empirical_ci(self, recommender):
        features = TaskFeatures(n_labeled=1000, target_f1=0.80)
        report = recommender.recommend(
            features, historical_points=HISTORICAL_POINTS, target_f1=0.80,
        )
        lo, hi = report.confidence_interval
        # CI should bracket the recommendation (roughly)
        assert lo > 0 or hi > 0  # at least one bound is non-zero

    def test_empirical_marginal_curve(self, recommender):
        features = TaskFeatures(n_labeled=1000, target_f1=0.80)
        report = recommender.recommend(
            features, historical_points=HISTORICAL_POINTS, target_f1=0.80,
        )
        assert len(report.marginal_returns_curve) > 0
        # Curve should be monotonically non-decreasing
        f1s = [f1 for _, f1 in report.marginal_returns_curve]
        assert all(f1s[i] <= f1s[i + 1] for i in range(len(f1s) - 1))

    def test_high_target_empirical(self, recommender):
        features = TaskFeatures(n_labeled=1000, target_f1=0.95)
        report = recommender.recommend(
            features, historical_points=HISTORICAL_POINTS, target_f1=0.95,
        )
        # Should still return a recommendation (even if target is ambitious)
        assert report.recommended_n > 0


# ---------------------------------------------------------------------------
# Cold-start mode
# ---------------------------------------------------------------------------


class TestColdStartMode:
    def test_basic_cold_start(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.80,
        )
        report = recommender.recommend(features)
        assert report.estimation_basis == "prior_based"
        assert report.prior_source is not None
        assert "HatEval" in report.prior_source
        assert report.update_after_n == 50
        assert report.recommended_n > 0

    def test_cold_start_3class(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=3, task_type="framing", target_f1=0.80,
        )
        report = recommender.recommend(features)
        assert "SemEval" in report.prior_source

    def test_cold_start_complex(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=10, task_type="moral", target_f1=0.70,
        )
        report = recommender.recommend(features)
        assert "MFTC" in report.prior_source

    def test_cold_start_has_curve(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.80,
        )
        report = recommender.recommend(features)
        assert len(report.marginal_returns_curve) > 0

    def test_cold_start_fitting_params(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.80,
        )
        report = recommender.recommend(features)
        assert report.fitting_params is not None
        assert report.fitting_params["b"] > 0  # exponent should be positive


# ---------------------------------------------------------------------------
# Fallback to cold-start with insufficient historical data
# ---------------------------------------------------------------------------


class TestFallback:
    def test_too_few_historical_points(self, recommender):
        """With <3 historical points, should fall back to cold-start."""
        features = TaskFeatures(
            n_labeled=50, n_classes=2, task_type="sentiment", target_f1=0.80,
        )
        report = recommender.recommend(
            features, historical_points=[(50, 0.55), (100, 0.63)],
        )
        assert report.estimation_basis == "prior_based"

    def test_none_historical(self, recommender):
        features = TaskFeatures(
            n_labeled=0, n_classes=3, task_type="topic", target_f1=0.75,
        )
        report = recommender.recommend(features, historical_points=None)
        assert report.estimation_basis == "prior_based"


# ---------------------------------------------------------------------------
# Report completeness
# ---------------------------------------------------------------------------


class TestReportCompleteness:
    @pytest.mark.parametrize("n_classes,task_type,target_f1", [
        (2, "sentiment", 0.80),
        (3, "framing", 0.75),
        (10, "moral", 0.70),
        (2, "stance", 0.85),
        (5, "topic", 0.80),
    ])
    def test_cold_start_all_fields(self, recommender, n_classes, task_type, target_f1):
        features = TaskFeatures(n_labeled=0, n_classes=n_classes, task_type=task_type, target_f1=target_f1)
        report = recommender.recommend(features)
        assert isinstance(report, BudgetReport)
        assert report.recommended_n > 0
        assert len(report.marginal_returns_curve) > 0
        assert report.estimation_basis == "prior_based"
        assert report.fitting_params is not None
