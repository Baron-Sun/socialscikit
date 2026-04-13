"""Tests for socialscikit.core.charts — visualization dashboard charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from socialscikit.core.charts import (
    format_annotation_stats_html,
    format_eval_metrics_html,
    format_review_stats_html,
    plot_annotation_progress,
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_per_class_metrics,
    plot_review_progress,
    plot_theme_distribution,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sample_per_class():
    return [
        {"label": "pos", "precision": 0.85, "recall": 0.90, "f1": 0.87, "support": 40},
        {"label": "neg", "precision": 0.78, "recall": 0.72, "f1": 0.75, "support": 30},
        {"label": "neu", "precision": 0.65, "recall": 0.60, "f1": 0.62, "support": 30},
    ]


@pytest.fixture
def sample_cm():
    return {
        "labels": ["pos", "neg", "neu"],
        "matrix": [
            [36, 2, 2],
            [5, 22, 3],
            [4, 8, 18],
        ],
    }


# ======================================================================
# Confusion matrix
# ======================================================================


class TestConfusionMatrix:
    def test_basic(self, sample_cm):
        fig = plot_confusion_matrix(sample_cm["labels"], sample_cm["matrix"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_class(self):
        fig = plot_confusion_matrix(["A"], [[10]])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_matrix(self):
        labels = [f"C{i}" for i in range(10)]
        matrix = [[5 if i == j else 1 for j in range(10)] for i in range(10)]
        fig = plot_confusion_matrix(labels, matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_english_labels(self, sample_cm):
        fig = plot_confusion_matrix(sample_cm["labels"], sample_cm["matrix"], lang="en")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Per-class metrics
# ======================================================================


class TestPerClassMetrics:
    def test_basic(self, sample_per_class):
        fig = plot_per_class_metrics(sample_per_class)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_per_class_metrics([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_class(self):
        data = [{"label": "X", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 10}]
        fig = plot_per_class_metrics(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_english(self, sample_per_class):
        fig = plot_per_class_metrics(sample_per_class, lang="en")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Label distribution
# ======================================================================


class TestLabelDistribution:
    def test_basic(self):
        true = {"pos": 40, "neg": 30, "neu": 30}
        pred = {"pos": 45, "neg": 25, "neu": 30}
        fig = plot_label_distribution(true, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_label_distribution({}, {})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_mismatched_labels(self):
        true = {"A": 10, "B": 20}
        pred = {"B": 15, "C": 15}
        fig = plot_label_distribution(true, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Annotation progress
# ======================================================================


class TestAnnotationProgress:
    def test_basic(self):
        fig = plot_annotation_progress(50, 5, 3, 42)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_done(self):
        fig = plot_annotation_progress(100, 0, 0, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_pending(self):
        fig = plot_annotation_progress(0, 0, 0, 100)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_annotation_progress(0, 0, 0, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Confidence histogram
# ======================================================================


class TestConfidenceHistogram:
    def test_basic(self):
        confs = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.6, 0.4]
        fig = plot_confidence_histogram(confs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_confidence_histogram([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_high(self):
        confs = [0.9, 0.95, 0.99, 0.88, 0.92]
        fig = plot_confidence_histogram(confs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Theme distribution
# ======================================================================


class TestThemeDistribution:
    def test_basic(self):
        themes = {"RQ1": 15, "RQ2": 8, "RQ3": 22}
        fig = plot_theme_distribution(themes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_theme_distribution({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single(self):
        fig = plot_theme_distribution({"Theme": 5})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_themes(self):
        themes = {f"T{i}": (i + 1) * 3 for i in range(12)}
        fig = plot_theme_distribution(themes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# Review progress
# ======================================================================


class TestReviewProgress:
    def test_basic(self):
        fig = plot_review_progress(10, 3, 2, 5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_done(self):
        fig = plot_review_progress(15, 2, 1, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_none_done(self):
        fig = plot_review_progress(0, 0, 0, 20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig = plot_review_progress(0, 0, 0, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ======================================================================
# HTML metric cards
# ======================================================================


class TestHTMLCards:
    def test_eval_metrics(self):
        html = format_eval_metrics_html(0.85, 0.82, 0.84, 0.78, 100, 85)
        assert "0.8500" in html
        assert "0.8200" in html

    def test_eval_metrics_english(self):
        html = format_eval_metrics_html(0.9, 0.88, 0.89, 0.85, 50, 45, lang="en")
        assert "Accuracy" in html
        assert "Macro F1" in html

    def test_review_stats(self):
        html = format_review_stats_html(20, 10, 3, 2, 5)
        assert "20" in html
        assert "10" in html

    def test_review_stats_english(self):
        html = format_review_stats_html(20, 10, 3, 2, 5, lang="en")
        assert "Accepted" in html
        assert "Pending" in html

    def test_annotation_stats(self):
        html = format_annotation_stats_html(100, 60, 5, 2, 33, 120.5)
        assert "100" in html
        assert "60" in html

    def test_annotation_stats_with_dist(self):
        dist = {"pos": 30, "neg": 20, "neu": 10}
        html = format_annotation_stats_html(100, 60, 5, 2, 33, 120.5, label_dist=dist)
        assert "100" in html
        # Distribution bar should be present
        assert "pos" in html or "neg" in html
