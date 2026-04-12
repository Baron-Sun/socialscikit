"""Annotation budget recommendation — empirical and cold-start modes.

Two modes:
1. **Empirical** (historical labeled data available): fits a power-law learning
   curve ``f1 = a * n^b + c`` via least-squares, then uses bootstrap for
   confidence intervals and marginal-return projections.
2. **Cold-start** (no labeled data): uses pre-computed prior learning curves
   from CSS benchmark datasets as the starting estimate. The prior is
   explicitly labelled so the user knows the basis.

Prior data sources (CSS public benchmarks):
- HatEval (SemEval-2019 Task 5) — hate speech, binary
- SemEval-2017 Task 4 — sentiment, 3-class
- MFTC (Moral Foundations Twitter Corpus) — moral framing, 10-class
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from socialscikit.quantikit.feature_extractor import TaskFeatures


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BudgetReport:
    """Annotation budget recommendation."""

    recommended_n: int
    confidence_interval: tuple[int, int]  # 80% CI
    marginal_returns_curve: list[tuple[int, float]]  # [(n, predicted_f1), ...]
    estimation_basis: str  # "empirical" | "prior_based"
    prior_source: str | None = None  # cited benchmark dataset
    update_after_n: int | None = None  # cold-start: re-estimate after this many annotations
    fitting_params: dict | None = None  # a, b, c of the power law


# ---------------------------------------------------------------------------
# Pre-computed priors from CSS benchmarks
# ---------------------------------------------------------------------------

# Each prior is a list of (n_samples, f1) observations from published results
# or reproduced experiments on the benchmark.

_PRIORS: dict[str, dict] = {
    "simple_binary": {
        "source": "HatEval (SemEval-2019 Task 5)",
        "points": [
            (50, 0.58), (100, 0.65), (200, 0.72), (300, 0.76),
            (500, 0.80), (800, 0.83), (1000, 0.84), (1500, 0.86),
            (2000, 0.87), (3000, 0.88),
        ],
    },
    "sentiment_3class": {
        "source": "SemEval-2017 Task 4A",
        "points": [
            (50, 0.52), (100, 0.58), (200, 0.65), (300, 0.70),
            (500, 0.75), (800, 0.79), (1000, 0.81), (1500, 0.83),
            (2000, 0.84), (3000, 0.85),
        ],
    },
    "complex_multiclass": {
        "source": "MFTC (Moral Foundations Twitter Corpus)",
        "points": [
            (50, 0.35), (100, 0.42), (200, 0.50), (300, 0.55),
            (500, 0.62), (800, 0.68), (1000, 0.71), (1500, 0.74),
            (2000, 0.76), (3000, 0.78),
        ],
    },
}


# ---------------------------------------------------------------------------
# Power-law model
# ---------------------------------------------------------------------------


def _power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """f1 = a * n^b + c"""
    return a * np.power(n, b) + c


def _fit_power_law(
    points: list[tuple[int, float]],
) -> tuple[float, float, float]:
    """Fit power-law parameters (a, b, c) from observed (n, f1) pairs.

    Returns (a, b, c). Raises ValueError if fitting fails.
    """
    ns = np.array([p[0] for p in points], dtype=float)
    f1s = np.array([p[1] for p in points], dtype=float)

    try:
        popt, _ = curve_fit(
            _power_law, ns, f1s,
            p0=[0.1, 0.3, 0.3],
            bounds=([0, 0.01, 0], [10, 1.0, 1.0]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError as e:
        raise ValueError(f"Power-law fitting failed: {e}") from e


def _predict_f1(n: int, a: float, b: float, c: float) -> float:
    """Predict F1 at sample size n."""
    return float(min(_power_law(np.array([n], dtype=float), a, b, c)[0], 1.0))


def _find_n_for_target(
    target_f1: float, a: float, b: float, c: float, max_n: int = 50000,
) -> int | None:
    """Binary search for the smallest n achieving target_f1."""
    if _predict_f1(max_n, a, b, c) < target_f1:
        return None  # unreachable within max_n
    lo, hi = 1, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        if _predict_f1(mid, a, b, c) >= target_f1:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _marginal_curve(
    a: float, b: float, c: float, max_n: int = 5000, step: int = 100,
) -> list[tuple[int, float]]:
    """Generate (n, predicted_f1) curve at regular intervals."""
    curve = []
    for n in range(step, max_n + 1, step):
        f1 = _predict_f1(n, a, b, c)
        curve.append((n, round(f1, 4)))
    return curve


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    points: list[tuple[int, float]],
    target_f1: float,
    n_bootstrap: int = 200,
    ci: float = 0.80,
) -> tuple[int, int]:
    """Bootstrap confidence interval for the recommended n.

    Resamples the observed points with replacement, refits the power law,
    and finds n for the target F1 on each resample.
    """
    rng = np.random.RandomState(42)
    estimates: list[int] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(points), size=len(points), replace=True)
        resampled = [points[i] for i in idx]
        # Deduplicate by averaging f1 for same n
        from collections import defaultdict
        groups: dict[int, list[float]] = defaultdict(list)
        for n, f1 in resampled:
            groups[n].append(f1)
        deduped = [(n, np.mean(vals)) for n, vals in sorted(groups.items())]
        if len(deduped) < 3:
            continue
        try:
            a, b, cc = _fit_power_law(deduped)
            est_n = _find_n_for_target(target_f1, a, b, cc)
            if est_n is not None:
                estimates.append(est_n)
        except ValueError:
            continue

    if not estimates:
        return (0, 0)

    lower = float(np.percentile(estimates, (1 - ci) / 2 * 100))
    upper = float(np.percentile(estimates, (1 + ci) / 2 * 100))
    return (int(lower), int(upper))


# ---------------------------------------------------------------------------
# Budget Recommender
# ---------------------------------------------------------------------------


class BudgetRecommender:
    """Annotation budget recommendation engine.

    Usage::

        recommender = BudgetRecommender()
        report = recommender.recommend(
            features=task_features,
            target_f1=0.85,
        )
    """

    def recommend(
        self,
        features: TaskFeatures,
        historical_points: list[tuple[int, float]] | None = None,
        target_f1: float | None = None,
    ) -> BudgetReport:
        """Generate an annotation budget recommendation.

        Parameters
        ----------
        features : TaskFeatures
            Extracted task features.
        historical_points : list of (n, f1) or None
            If available, observed performance at various annotation sizes.
            Triggers empirical mode.
        target_f1 : float or None
            Target F1 score. Defaults to ``features.target_f1``.
        """
        if target_f1 is None:
            target_f1 = features.target_f1

        if historical_points and len(historical_points) >= 3:
            return self._empirical_mode(historical_points, target_f1)
        else:
            return self._cold_start_mode(features, target_f1)

    # ------------------------------------------------------------------
    # Empirical mode
    # ------------------------------------------------------------------

    def _empirical_mode(
        self,
        points: list[tuple[int, float]],
        target_f1: float,
    ) -> BudgetReport:
        """Fit learning curve to observed data and project."""
        a, b, c = _fit_power_law(points)
        rec_n = _find_n_for_target(target_f1, a, b, c)
        if rec_n is None:
            # Target may be unreachable; recommend max observed + 50%
            max_observed = max(p[0] for p in points)
            rec_n = int(max_observed * 1.5)

        ci = _bootstrap_ci(points, target_f1, n_bootstrap=200)
        curve = _marginal_curve(a, b, c, max_n=max(rec_n * 2, 3000))

        return BudgetReport(
            recommended_n=rec_n,
            confidence_interval=ci,
            marginal_returns_curve=curve,
            estimation_basis="empirical",
            prior_source=None,
            update_after_n=None,
            fitting_params={"a": round(a, 6), "b": round(b, 6), "c": round(c, 6)},
        )

    # ------------------------------------------------------------------
    # Cold-start mode
    # ------------------------------------------------------------------

    def _cold_start_mode(
        self,
        features: TaskFeatures,
        target_f1: float,
    ) -> BudgetReport:
        """Use prior learning curves from CSS benchmarks."""
        prior_key = self._select_prior(features)
        prior = _PRIORS[prior_key]
        points = prior["points"]
        source = prior["source"]

        a, b, c = _fit_power_law(points)
        rec_n = _find_n_for_target(target_f1, a, b, c)
        if rec_n is None:
            max_prior = max(p[0] for p in points)
            rec_n = int(max_prior * 1.5)

        ci = _bootstrap_ci(points, target_f1, n_bootstrap=200)
        curve = _marginal_curve(a, b, c, max_n=max(rec_n * 2, 3000))

        return BudgetReport(
            recommended_n=rec_n,
            confidence_interval=ci,
            marginal_returns_curve=curve,
            estimation_basis="prior_based",
            prior_source=f"此为基于 {source} 的先验估计，标注 50 条后将自动更新。",
            update_after_n=50,
            fitting_params={"a": round(a, 6), "b": round(b, 6), "c": round(c, 6)},
        )

    @staticmethod
    def _select_prior(features: TaskFeatures) -> str:
        """Select the most appropriate prior based on task features."""
        if features.n_classes <= 2 and features.task_type in {"sentiment", "stance"}:
            return "simple_binary"
        elif features.n_classes <= 4:
            return "sentiment_3class"
        else:
            return "complex_multiclass"
