"""Tests for the ICR (Inter-Coder Reliability) module."""

import pytest

from socialscikit.core.icr import ICRCalculator, ICRReport, ICRResult, PerCategoryAgreement


@pytest.fixture
def calc():
    return ICRCalculator()


# =====================================================================
# Cohen's Kappa
# =====================================================================


class TestCohensKappa:
    def test_perfect_agreement(self, calc):
        labels = ["pos", "neg", "pos", "neg", "pos"]
        result = calc.compute_cohens_kappa(labels, labels)
        assert result.metric_name == "cohens_kappa"
        assert result.value == 1.0
        assert "perfect" in result.interpretation.lower()

    def test_no_agreement(self, calc):
        c1 = ["pos", "pos", "pos", "pos"]
        c2 = ["neg", "neg", "neg", "neg"]
        result = calc.compute_cohens_kappa(c1, c2)
        assert result.value <= 0.0

    def test_moderate_agreement(self, calc):
        c1 = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        c2 = ["pos", "neg", "pos", "pos", "pos", "neg", "neg", "neg", "pos", "neg"]
        result = calc.compute_cohens_kappa(c1, c2)
        assert 0.0 < result.value < 1.0
        assert result.n_items == 10
        assert result.n_categories == 2

    def test_empty_input(self, calc):
        result = calc.compute_cohens_kappa([], [])
        assert result.value == 0.0
        assert result.n_items == 0

    def test_length_mismatch(self, calc):
        with pytest.raises(ValueError, match="Length mismatch"):
            calc.compute_cohens_kappa(["a", "b"], ["a"])

    def test_single_class(self, calc):
        c1 = ["pos", "pos", "pos"]
        c2 = ["pos", "pos", "pos"]
        result = calc.compute_cohens_kappa(c1, c2)
        # When all items are the same class, p_e = 1 and kappa is 0 by convention
        # But our implementation returns 0.0 when p_e >= 1
        assert result.value >= 0.0

    def test_multiclass(self, calc):
        c1 = ["a", "b", "c", "a", "b", "c"]
        c2 = ["a", "b", "c", "b", "a", "c"]
        result = calc.compute_cohens_kappa(c1, c2)
        assert result.n_categories == 3
        assert 0.0 < result.value < 1.0

    def test_explicit_labels(self, calc):
        c1 = ["pos", "neg"]
        c2 = ["pos", "pos"]
        result = calc.compute_cohens_kappa(c1, c2, labels=["pos", "neg", "neutral"])
        assert result.n_categories == 3


# =====================================================================
# Krippendorff's Alpha
# =====================================================================


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self, calc):
        matrix = [
            ["a", "a"],
            ["b", "b"],
            ["c", "c"],
        ]
        result = calc.compute_krippendorffs_alpha(matrix)
        assert result.metric_name == "krippendorffs_alpha"
        assert result.value == 1.0
        assert "reliable" in result.interpretation.lower()

    def test_no_agreement(self, calc):
        # Systematically opposing: coder1=a when coder2=b and vice versa
        matrix = [
            ["a", "b"],
            ["b", "a"],
            ["a", "b"],
            ["b", "a"],
        ]
        result = calc.compute_krippendorffs_alpha(matrix)
        assert result.value < 0.667  # unreliable

    def test_with_missing_values(self, calc):
        matrix = [
            ["a", "a", None],
            ["b", None, "b"],
            ["a", "a", "a"],
            [None, "b", "b"],
        ]
        result = calc.compute_krippendorffs_alpha(matrix)
        assert result.n_coders == 3
        assert result.n_items > 0

    def test_empty_matrix(self, calc):
        result = calc.compute_krippendorffs_alpha([])
        assert result.value == 0.0

    def test_single_value(self, calc):
        matrix = [["a", "a"], ["a", "a"]]
        result = calc.compute_krippendorffs_alpha(matrix)
        assert result.value == 1.0

    def test_three_coders(self, calc):
        matrix = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["a", "a", "b"],
            ["b", "a", "b"],
        ]
        result = calc.compute_krippendorffs_alpha(matrix)
        assert result.n_coders == 3
        assert 0.0 < result.value < 1.0

    def test_unsupported_data_type(self, calc):
        with pytest.raises(ValueError, match="nominal"):
            calc.compute_krippendorffs_alpha([["a", "b"]], data_type="interval")


# =====================================================================
# Jaccard Agreement (Multi-label)
# =====================================================================


class TestMultilabelAgreement:
    def test_identical_sets(self, calc):
        s1 = [{"a", "b"}, {"c"}, {"a", "b", "c"}]
        s2 = [{"a", "b"}, {"c"}, {"a", "b", "c"}]
        result = calc.compute_multilabel_agreement(s1, s2)
        assert result.value == 1.0

    def test_disjoint_sets(self, calc):
        s1 = [{"a", "b"}, {"c"}]
        s2 = [{"c", "d"}, {"a"}]
        result = calc.compute_multilabel_agreement(s1, s2)
        assert result.value == 0.0

    def test_partial_overlap(self, calc):
        s1 = [{"a", "b", "c"}]
        s2 = [{"a", "b"}]
        result = calc.compute_multilabel_agreement(s1, s2)
        # Jaccard = |{a,b}| / |{a,b,c}| = 2/3
        assert abs(result.value - 2 / 3) < 0.01

    def test_both_empty(self, calc):
        s1 = [set(), set()]
        s2 = [set(), set()]
        result = calc.compute_multilabel_agreement(s1, s2)
        # Both empty = perfect agreement
        assert result.value == 1.0

    def test_one_empty(self, calc):
        s1 = [{"a", "b"}]
        s2 = [set()]
        result = calc.compute_multilabel_agreement(s1, s2)
        assert result.value == 0.0

    def test_empty_input(self, calc):
        result = calc.compute_multilabel_agreement([], [])
        assert result.value == 0.0

    def test_length_mismatch(self, calc):
        with pytest.raises(ValueError, match="Length mismatch"):
            calc.compute_multilabel_agreement([{"a"}], [{"a"}, {"b"}])


# =====================================================================
# Comprehensive Reports
# =====================================================================


class TestComputeAll:
    def test_returns_icr_report(self, calc):
        c1 = ["pos", "neg", "pos", "neg"]
        c2 = ["pos", "neg", "neg", "neg"]
        report = calc.compute_all(c1, c2)
        assert isinstance(report, ICRReport)
        assert len(report.results) == 2  # kappa + alpha
        assert report.results[0].metric_name == "cohens_kappa"
        assert report.results[1].metric_name == "krippendorffs_alpha"
        assert len(report.per_category) == 2  # pos + neg
        assert report.summary_text  # not empty

    def test_per_category_populated(self, calc):
        c1 = ["a", "b", "c", "a", "b"]
        c2 = ["a", "b", "b", "a", "c"]
        report = calc.compute_all(c1, c2)
        assert len(report.per_category) == 3
        for pc in report.per_category:
            assert isinstance(pc, PerCategoryAgreement)
            assert 0.0 <= pc.observed_agreement <= 1.0
            assert 0.0 <= pc.specific_agreement <= 1.0


class TestComputeAllMultilabel:
    def test_returns_report(self, calc):
        s1 = [{"a", "b"}, {"c"}, {"a"}]
        s2 = [{"a"}, {"c", "d"}, {"a", "b"}]
        report = calc.compute_all_multilabel(s1, s2)
        assert isinstance(report, ICRReport)
        assert len(report.results) == 2  # jaccard + avg_per_theme_kappa
        assert report.results[0].metric_name == "jaccard_agreement"
        assert report.results[1].metric_name == "avg_per_theme_kappa"
        assert report.summary_text


# =====================================================================
# Interpretation
# =====================================================================


class TestInterpretation:
    def test_kappa_scale(self, calc):
        assert "poor" in calc.interpret_kappa(-0.1).lower()
        assert "slight" in calc.interpret_kappa(0.1).lower()
        assert "fair" in calc.interpret_kappa(0.3).lower()
        assert "moderate" in calc.interpret_kappa(0.5).lower()
        assert "substantial" in calc.interpret_kappa(0.7).lower()
        assert "perfect" in calc.interpret_kappa(0.9).lower()

    def test_alpha_scale(self, calc):
        assert "discard" in calc.interpret_alpha(0.5).lower()
        assert "tentative" in calc.interpret_alpha(0.7).lower()
        assert "reliable" in calc.interpret_alpha(0.9).lower()


# =====================================================================
# Format Report
# =====================================================================


class TestFormatReport:
    def test_format_zh(self, calc):
        c1 = ["a", "b", "a"]
        c2 = ["a", "a", "a"]
        report = calc.compute_all(c1, c2)
        text = report.summary_text
        assert "信度" in text or "Kappa" in text

    def test_format_en(self, calc):
        c1 = ["a", "b"]
        c2 = ["a", "b"]
        report = calc.compute_all(c1, c2)
        text = calc.format_report(report, lang="en")
        assert "Inter-Coder" in text
